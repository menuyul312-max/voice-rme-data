"""Fail-closed benchmark verdict. No benchmark alone certifies enterprise release."""
REQUIRED = frozenset(f'G{i}' for i in range(1, 8))


def release_verdict(gates):
    ids = [g['gerbang'] for g in gates]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate gate IDs')
    by_id = {g['gerbang']: g['status'] for g in gates}
    failed = sorted(k for k, v in by_id.items() if v == 'GAGAL')
    if failed:
        return 'BELUM LAYAK RILIS: ' + ', '.join(failed) + ' gagal'
    missing = sorted(k for k in REQUIRED if by_id.get(k) != 'LULUS')
    if missing:
        return 'BELUM DAPAT DIPUTUSKAN: data kurang untuk ' + ', '.join(missing)
    return ('LULUS GERBANG BENCHMARK; kandidat uji klinis terkendali. '
            'Rilis enterprise menunggu validasi klinis, keamanan, operasional dan integrasi.')
