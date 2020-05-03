def get_price_uri(crypto, currency):
    if isinstance(crypto, list):
        fsyms = ",".join(crypto)
    else:
        fsyms = crypto
    return f'https://min-api.cryptocompare.com/data/pricemulti?fsyms={fsyms}&tsyms={currency}'