"""
Global No-op I/O Module

This module provides functions for globally silencing I/O operations during training,
especially for production runs or debugging scenarios where visualization output
should be suppressed.

Functions:
- install_noop_sinks: Replace common I/O functions with no-op versions

This module is designed to be imported early in the training pipeline to suppress
unwanted file operations and console output.
"""

import os
import builtins
import types


def install_noop_sinks(mute_print=False):
    """
    Global silent mode: Replace common image saving and plotting functions with no-op.
    Does not change training/loss, only suppresses visualization output and redundant prints.
    
    Args:
        mute_print: Whether to also mute console print statements (default: False)
        
    Returns:
        None
        
    Side effects:
        - Replaces matplotlib.pyplot.savefig, plt.clf, plt.close with no-op
        - Replaces cv2.imwrite, imageio.imwrite, torchvision.utils.save_image with no-op
        - Replaces PIL.Image.save with no-op (both class method and instance methods)
        - Optionally replaces builtins.print with quiet version
    """
    print("[noop_io] Installing global no-op sinks...")
    
    # 1) Image saving functions → no-op
    try:
        import matplotlib.pyplot as plt
        def _nofig(*args, **kwargs): 
            return None
        plt.savefig = _nofig
        plt.clf = _nofig
        plt.close = _nofig
        print("[noop_io] matplotlib.pyplot functions replaced with no-op")
    except Exception as e:
        print(f"[noop_io] Failed to replace matplotlib functions: {e}")
    
    # Replace other image saving functions
    for mod_name, fn_name in [
        ("cv2", "imwrite"),
        ("imageio", "imwrite"),
        ("torchvision.utils", "save_image"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            if hasattr(mod, fn_name):
                setattr(mod, fn_name, lambda *a, **k: False)
                print(f"[noop_io] {mod_name}.{fn_name} replaced with no-op")
        except Exception as e:
            print(f"[noop_io] Failed to replace {mod_name}.{fn_name}: {e}")
    
    # Special handling for PIL.Image.save
    try:
        from PIL import Image
        
        def _noop_save(self, *args, **kwargs):
            return None
        
        # Replace class method
        Image.save = _noop_save
        
        # Replace all existing instance methods
        for obj in list(Image.__dict__.values()):
            if hasattr(obj, 'save'):
                obj.save = types.MethodType(_noop_save, obj)
        
        print("[noop_io] PIL.Image.save replaced with no-op")
    except Exception as e:
        print(f"[noop_io] Failed to replace PIL.Image.save: {e}")
    
    # 2) Optional: Mute console output
    if mute_print:
        try:
            builtins_print = builtins.print
            
            def _quiet_print(*args, **kwargs):
                # Allow critical warnings to pass through
                txt = " ".join(map(str, args))
                if "[WARN]" in txt or "[ERROR]" in txt or "[CRITICAL]" in txt:
                    return builtins_print(*args, **kwargs)
                # Mute everything else
                return None
            
            builtins.print = _quiet_print
            print("[noop_io] Console output muted (warnings/errors still shown)")
        except Exception as e:
            print(f"[noop_io] Failed to mute console output: {e}")
    
    print("[noop_io] Global no-op sinks installation complete")


def restore_original_io():
    """
    Restore original I/O functions (for debugging purposes).
    
    Note: This function is mainly for debugging and may not fully restore
    all original functionality due to the way Python imports work.
    
    Returns:
        None
    """
    print("[noop_io] Restoring original I/O functions...")
    
    # Note: Full restoration is complex due to import mechanisms
    # This is mainly for debugging purposes
    try:
        # Restore print function
        import builtins
        if hasattr(builtins, '_original_print'):
            builtins.print = builtins._original_print
            print("[noop_io] Print function restored")
    except Exception as e:
        print(f"[noop_io] Failed to restore original I/O: {e}")
    
    print("[noop_io] I/O restoration complete (partial)")


def check_noop_status():
    """
    Check the status of no-op replacements.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "matplotlib": False,
        "cv2": False,
        "imageio": False,
        "torchvision": False,
        "pil": False,
        "print": False
    }
    
    try:
        import matplotlib.pyplot as plt
        status["matplotlib"] = plt.savefig.__name__ == "_nofig"
    except:
        pass
    
    try:
        import cv2
        status["cv2"] = cv2.imwrite.__name__ == "<lambda>"
    except:
        pass
    
    try:
        import imageio
        status["imageio"] = imageio.imwrite.__name__ == "<lambda>"
    except:
        pass
    
    try:
        import torchvision.utils
        status["torchvision"] = torchvision.utils.save_image.__name__ == "<lambda>"
    except:
        pass
    
    try:
        from PIL import Image
        status["pil"] = Image.save.__name__ == "_noop_save"
    except:
        pass
    
    try:
        import builtins
        status["print"] = builtins.print.__name__ == "_quiet_print"
    except:
        pass
    
    return status