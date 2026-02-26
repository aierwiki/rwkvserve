"""
Data-Parallel Inference Engine - Multi-GPU Data Parallelism

Manages multiple GPU workers behind a single API entry point with automatic
round-robin load balancing. Each worker runs an independent model instance
with its own State Cache.
"""

import os
import asyncio
import torch
import multiprocessing as mp
from multiprocessing import Queue, Event
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
import traceback
import uuid
import time


@dataclass
class WorkerRequest:
    """Request sent to a worker process."""
    request_id: str
    method: str  # "generate", "generate_stream", "shutdown"
    kwargs: Dict[str, Any]


@dataclass
class WorkerResponse:
    """Response returned from a worker process."""
    request_id: str
    success: bool
    result: Any = None
    error: str = ""
    is_stream_chunk: bool = False
    is_stream_end: bool = False


def _worker_process(
    rank: int,
    gpu_id: int,
    model_path: str,
    model_name: str,
    dtype: Optional[str],
    max_batch_size: int,
    prefill_chunk_size: int,
    cache_config: Dict[str, Any],
    request_queue: Queue,
    response_queue: Queue,
    ready_event: Event,
):
    """
    Worker process entry point.

    Runs in a separate process bound to a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from rwkvserve.models import AutoModel
    from rwkvserve.data.tokenizers import AutoTokenizer
    from rwkvserve.engine.core import RWKVEngine
    from rwkvserve.inference.state_cache import create_state_cache
    from rwkvserve.api.async_serving_chat import AsyncChatCompletionService
    
    print(f"[Worker {rank}] Starting, bound to GPU {gpu_id}...")
    
    try:
        print(f"[Worker {rank}] Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, device="cuda", dtype=dtype)
        model.eval()
        print(f"[Worker {rank}] Model loaded")
        
        state_cache = None
        if cache_config.get("cache_level", "prefix") != "none":
            try:
                state_cache = create_state_cache(
                    model=model,
                    max_cache_memory_gb=cache_config.get("max_cache_memory_gb", 4.0),
                    cache_level=cache_config.get("cache_level", "prefix"),
                    cache_first_message=cache_config.get("cache_first_message", True),
                )
                print(f"[Worker {rank}] State Cache initialized")
            except Exception as e:
                print(f"[Worker {rank}] State Cache init failed: {e}")
        
        engine = RWKVEngine(
            model,
            tokenizer,
            max_batch_size=max_batch_size,
            prefill_chunk_size=prefill_chunk_size,
            state_cache=state_cache,
        )
        chat_service = AsyncChatCompletionService(
            engine=engine,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            state_cache=state_cache,
        )
        
        ready_event.set()
        print(f"[Worker {rank}] Ready, waiting for requests...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def handle_request(req: WorkerRequest):
            from rwkvserve.api.protocol import ChatCompletionRequest
            
            try:
                if req.method == "shutdown":
                    return WorkerResponse(
                        request_id=req.request_id,
                        success=True,
                        result="shutdown"
                    )
                
                elif req.method == "generate":
                    chat_request = ChatCompletionRequest(**req.kwargs)
                    chat_request.stream = False
                    
                    result = await chat_service.create_chat_completion(chat_request)
                    
                    if hasattr(result, "model_dump"):
                        result = result.model_dump()
                    elif hasattr(result, "dict"):
                        result = result.dict()
                    
                    return WorkerResponse(
                        request_id=req.request_id,
                        success=True,
                        result=result
                    )
                
                elif req.method == "generate_stream":
                    chat_request = ChatCompletionRequest(**req.kwargs)
                    chat_request.stream = True
                    
                    result = await chat_service.create_chat_completion(chat_request)
                    
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            if hasattr(chunk, "model_dump"):
                                chunk = chunk.model_dump()
                            elif hasattr(chunk, "dict"):
                                chunk = chunk.dict()
                            
                            response_queue.put(WorkerResponse(
                                request_id=req.request_id,
                                success=True,
                                result=chunk,
                                is_stream_chunk=True,
                            ))
                    
                    return WorkerResponse(
                        request_id=req.request_id,
                        success=True,
                        is_stream_end=True,
                    )
                
                else:
                    return WorkerResponse(
                        request_id=req.request_id,
                        success=False,
                        error=f"Unknown method: {req.method}"
                    )
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                return WorkerResponse(
                    request_id=req.request_id,
                    success=False,
                    error=str(e)
                )
        
        def run_loop():
            while True:
                try:
                    req = request_queue.get(timeout=1.0)
                    
                    if req.method == "shutdown":
                        print(f"[Worker {rank}] Received shutdown signal")
                        response_queue.put(WorkerResponse(
                            request_id=req.request_id,
                            success=True,
                            result="shutdown"
                        ))
                        break
                    
                    response = loop.run_until_complete(handle_request(req))
                    response_queue.put(response)
                    
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        print(f"[Worker {rank}] Error: {e}")
        
        run_loop()
        
    except Exception as e:
        print(f"[Worker {rank}] Startup failed: {e}")
        traceback.print_exc()
        ready_event.set()
    
    print(f"[Worker {rank}] Exited")


class DataParallelEngine:
    """
    Data-parallel inference engine.

    Manages multiple GPU workers and provides a unified inference interface
    with round-robin load balancing.
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        gpu_ids: List[int],
        dtype: Optional[str] = None,
        max_batch_size: int = 1,
        prefill_chunk_size: int = 512,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.num_workers = len(gpu_ids)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.cache_config = cache_config or {
            "max_cache_memory_gb": 4.0,
            "cache_level": "prefix",
            "cache_first_message": True,
        }
        
        self.workers: List[mp.Process] = []
        self.request_queues: List[Queue] = []
        self.response_queue: Queue = mp.Queue()
        self.ready_events: List[Event] = []
        
        self._next_worker: int = 0
        self._request_lock = asyncio.Lock()
        
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        
        self._response_handler_task: Optional[asyncio.Task] = None
        self._running = False
        
        print(f"\n{'='*60}")
        print(f"  Data-Parallel Engine Init")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"GPUs: {gpu_ids} ({self.num_workers} workers)")
        print(f"State Cache: {cache_config.get('cache_level', 'prefix')}")
        print(f"{'='*60}\n")
    
    async def start(self) -> bool:
        """
        Start all worker processes.
        
        Returns:
            True if all workers started successfully.
        """
        print("Starting worker processes...")
        
        ctx = mp.get_context("spawn")
        
        for rank, gpu_id in enumerate(self.gpu_ids):
            request_queue = ctx.Queue()
            ready_event = ctx.Event()
            
            p = ctx.Process(
                target=_worker_process,
                args=(
                    rank,
                    gpu_id,
                    self.model_path,
                    self.model_name,
                    self.dtype,
                    self.max_batch_size,
                    self.prefill_chunk_size,
                    self.cache_config,
                    request_queue,
                    self.response_queue,
                    ready_event,
                ),
            )
            p.start()
            
            self.workers.append(p)
            self.request_queues.append(request_queue)
            self.ready_events.append(ready_event)
            
            print(f"  Worker {rank}: GPU {gpu_id} (PID: {p.pid})")
        
        print("\nWaiting for workers to load models...")
        timeout = 300
        start_time = time.time()
        
        all_ready = True
        for rank, event in enumerate(self.ready_events):
            remaining = max(0, timeout - (time.time() - start_time))
            if not event.wait(timeout=remaining):
                print(f"  Worker {rank} timed out")
                all_ready = False
            else:
                elapsed = time.time() - start_time
                print(f"  Worker {rank} ready ({elapsed:.1f}s)")
        
        if all_ready:
            print(f"\nAll {self.num_workers} workers ready")
            self._running = True
            
            self._response_handler_task = asyncio.create_task(self._response_handler())
            
            return True
        else:
            print("\nSome workers failed to start")
            await self.shutdown()
            return False
    
    async def _response_handler(self):
        """Background task: process worker responses."""
        while self._running:
            try:
                try:
                    response = self.response_queue.get_nowait()
                except:
                    await asyncio.sleep(0.001)
                    continue
                
                request_id = response.request_id
                
                if response.is_stream_chunk:
                    if request_id in self._stream_queues:
                        await self._stream_queues[request_id].put(response.result)
                
                elif response.is_stream_end:
                    if request_id in self._stream_queues:
                        await self._stream_queues[request_id].put(None)
                        del self._stream_queues[request_id]
                
                else:
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if response.success:
                            future.set_result(response.result)
                        else:
                            future.set_exception(Exception(response.error))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Response handler error: {e}")
    
    async def _select_worker(self) -> int:
        """Select next worker using round-robin."""
        async with self._request_lock:
            worker_id = self._next_worker
            self._next_worker = (self._next_worker + 1) % self.num_workers
            return worker_id
    
    async def generate(self, **kwargs) -> Any:
        """Non-streaming generation with round-robin worker selection."""
        request_id = str(uuid.uuid4())
        worker_id = await self._select_worker()
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_requests[request_id] = future
        
        req = WorkerRequest(
            request_id=request_id,
            method="generate",
            kwargs=kwargs,
        )
        self.request_queues[worker_id].put(req)
        
        result = await future
        return result
    
    async def generate_stream(self, **kwargs) -> AsyncGenerator:
        """Streaming generation, yields results from the assigned worker."""
        request_id = str(uuid.uuid4())
        worker_id = await self._select_worker()
        
        stream_queue = asyncio.Queue()
        self._stream_queues[request_id] = stream_queue
        
        req = WorkerRequest(
            request_id=request_id,
            method="generate_stream",
            kwargs=kwargs,
        )
        self.request_queues[worker_id].put(req)
        
        try:
            while True:
                chunk = await stream_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if request_id in self._stream_queues:
                del self._stream_queues[request_id]
    
    async def shutdown(self):
        """Shut down all workers."""
        print("\nShutting down data-parallel engine...")
        
        self._running = False
        
        if self._response_handler_task:
            self._response_handler_task.cancel()
            try:
                await self._response_handler_task
            except asyncio.CancelledError:
                pass
        
        for i, queue in enumerate(self.request_queues):
            try:
                queue.put(WorkerRequest(
                    request_id="shutdown",
                    method="shutdown",
                    kwargs={},
                ))
            except:
                pass
        
        for p in self.workers:
            try:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
            except:
                pass
        
        print("Data-parallel engine shut down")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "num_workers": self.num_workers,
            "gpu_ids": self.gpu_ids,
            "load_balance_strategy": "round_robin",
            "next_worker": self._next_worker,
            "pending_requests": len(self._pending_requests),
            "stream_requests": len(self._stream_queues),
        }
