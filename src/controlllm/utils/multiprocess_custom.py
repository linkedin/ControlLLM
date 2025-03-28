import multiprocess.pool as mp_pool
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
TERMINATE = mp_pool.TERMINATE  # module constant


# Define a custom _terminate_pool method for the Pool class.
def custom_terminate_pool(cls, taskqueue, inqueue, outqueue, pool, change_notifier,
                          worker_handler, task_handler, result_handler, cache):
    logging.debug('Custom _terminate_pool: finalizing pool')
    
    # Notify that worker_handler is terminated and unblock _handle_workers loop.
    worker_handler._state = TERMINATE
    change_notifier.put(None)
    
    # Mark the task handler as terminated.
    task_handler._state = TERMINATE
    logging.debug('Custom _terminate_pool: helping task handler/workers to finish')
    cls._help_stuff_finish(inqueue, task_handler, len(pool))
    
    # Ensure result handler is still alive if cache exists.
    if (not result_handler.is_alive()) and (len(cache) != 0):
        raise AssertionError("Cannot have cache with result_handler not alive")
    
    # Terminate the result handler and send the sentinel.
    result_handler._state = TERMINATE
    change_notifier.put(None)
    outqueue.put(None)  # sentinel
    
    # Wait for the worker handler to exit.
    logging.debug('Custom _terminate_pool: joining worker handler')
    if threading.current_thread() is not worker_handler:
        worker_handler.join(timeout=30)  # join with timeout
    
    # Terminate any worker that hasn't finished.
    if pool and hasattr(pool[0], 'terminate'):
        logging.debug('Custom _terminate_pool: terminating workers')
        for p in pool:
            if p.exitcode is None:
                p.terminate()
    
    # Join task handler with timeout.
    logging.debug('Custom _terminate_pool: joining task handler')
    if threading.current_thread() is not task_handler:
        task_handler.join(timeout=10)
    
    # Join result handler with timeout.
    logging.debug('Custom _terminate_pool: joining result handler')
    if threading.current_thread() is not result_handler:
        result_handler.join(timeout=10)
    
    # Join each pool worker with a timeout.
    if pool and hasattr(pool[0], 'terminate'):
        logging.debug('Custom _terminate_pool: joining pool workers with timeout')
        for p in pool:
            if p.is_alive():
                logging.debug('Custom _terminate_pool: cleaning up worker %d', p.pid)
                p.join(timeout=10)
    
    logging.debug('Custom _terminate_pool: termination finished.')

# Apply monkey patch early
mp_pool.Pool._terminate_pool = classmethod(custom_terminate_pool)


import sys
import datasets.fingerprint as fp
from datasets.fingerprint import update_fingerprint as original_update_fingerprint

def patched_update_fingerprint(fingerprint, transform, transform_args):
    # (Your patched implementation here; see previous code snippet.)
    # This example uses the fallback to to_string() for dict parameters containing a model.
    import logging
    from datasets.fingerprint import generate_random_fingerprint, _CACHING_ENABLED, fingerprint_warnings, Hasher

    logger = logging.getLogger(__name__)
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except Exception:
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, a random hash was used instead. "
                    "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                    "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                    "This warning is only showed once. Subsequent hashing failures won't be showed."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead.")
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
            )
        return generate_random_fingerprint()

    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except Exception:
            param = transform_args[key]
            if isinstance(param, dict) and "model" in param:
                safe_param = dict(param)
                try:
                    safe_param["model"] = safe_param["model"].to_string()
                except Exception:
                    safe_param["model"] = str(safe_param["model"])
                try:
                    hasher.update(safe_param)
                    continue
                except Exception:
                    pass
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, even after attempting to use to_string(), a random hash was used instead. "
                        "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                        "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                        "This warning is only showed once. Subsequent hashing failures won't be showed."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, even after attempting to use to_string(), a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, even after attempting to use to_string(), a random hash was used instead. This doesn't affect caching since it's disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()

# Patch the fingerprint module
fp.update_fingerprint = patched_update_fingerprint

# Optionally, iterate through sys.modules to patch any module-level copies of update_fingerprint.
for mod in sys.modules.values():
    if hasattr(mod, "update_fingerprint"):
        attr = getattr(mod, "update_fingerprint")
        if attr is original_update_fingerprint:
            setattr(mod, "update_fingerprint", patched_update_fingerprint)
