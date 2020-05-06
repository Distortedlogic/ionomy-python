from typing import Any

def currency_to_crypto():
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs):
            rt = func(*args, **kwargs)
            if isinstance(rt, list):
                final = []
                for item in rt:
                    item['crypto'] = item['currency']
                    del item['currency']
                    final.append(item)
                return final
            if isinstance(rt, dict):
                rt['crypto'] = rt['currency']
                del rt['currency']
                try:
                    withdrawls = rt['withdrawals']
                    final = []
                    for item in withdrawls:
                        item['crypto'] = item['currency']
                        del item['currency']
                        final.append(item)
                    rt['withdrawals'] = final
                except:
                    pass
                return rt
        return wrapper
    return decorator