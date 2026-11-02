"""Conservative numeric evidence matching; unsupported/ambiguous values stay absent.
This is a precision-first draft guard, not a clinical truth validator. Number words,
negated/historical contexts require human review and may be rejected.
"""
import re
import math

PATTERNS = {
    'blood_pressure': r'\b(?:tensi|tekanan darah|td|bp)\s*[:=]?\s*(\d{2,3})\s*(?:/|per)\s*(\d{2,3})(?![\d.,])',
    'heart_rate': r'\b(?:nadi|hr|heart rate|detak jantung)\s*[:=]?\s*(\d{2,3})(?![\d.,])',
    'respiratory_rate': r'\b(?:rr|frekuensi napas|frekuensi pernapasan|laju napas)\s*[:=]?\s*(\d{1,2})(?![\d.,])',
    'temperature': r'\b(?:suhu|temperatur|temp)\s*[:=]?\s*(\d{2}(?:[.,]\d{1,2})?)(?![\d.,])',
    'spo2': r'\b(?:spo2|saturasi)\s*[:=]?\s*(\d{2,3})(?![\d.,])',
}
RANGES = {'systolic': (40, 300), 'diastolic': (20, 200), 'heart_rate': (20, 250),
          'respiratory_rate': (4, 80), 'temperature': (25, 45), 'spo2': (50, 100)}


def extract_supported_vitals(text):
    """Require named measurement, unique value, and plausible range; reject ambiguity."""
    found = {k: set() for k in PATTERNS}
    # Sentence/turn boundary; do not split decimal punctuation.
    for clause in re.split(r'[;!?\n]|(?<!\d)\.(?!\d)', text.lower()):
        if re.search(r'\b(tidak|bukan|belum|jangan|riwayat|kemarin|contoh|misalnya|target)\b', clause):
            continue
        for key, pattern in PATTERNS.items():
            for m in re.finditer(pattern, clause):
                found[key].add(tuple(float(x.replace(',', '.')) for x in m.groups()))
    out = {}
    for key, values in found.items():
        if len(values) != 1:
            continue
        value = next(iter(values))
        if key == 'blood_pressure':
            sys_, dia = value
            if RANGES['systolic'][0] <= sys_ <= RANGES['systolic'][1] and RANGES['diastolic'][0] <= dia <= RANGES['diastolic'][1] and sys_ > dia:
                out.update(systolic=int(sys_), diastolic=int(dia))
        elif RANGES[key][0] <= value[0] <= RANGES[key][1]:
            out[key] = value[0] if key == 'temperature' else int(value[0])
    return out


def filter_vitals(proposed, transcript):
    supported = extract_supported_vitals(transcript)
    accepted, rejected = {}, []
    for key in RANGES:
        value = proposed.get(key)
        if value is None:
            continue
        try:
            number = float(str(value).replace(',', '.'))
            valid = not isinstance(value, bool) and math.isfinite(number) and number == supported.get(key)
        except (ValueError, TypeError, OverflowError):
            valid = False
        if valid:
            accepted[key] = supported[key]
        else:
            rejected.append(key)
    # Blood pressure is an atomic pair.
    if ('systolic' in accepted) != ('diastolic' in accepted):
        for key in ('systolic', 'diastolic'):
            if key in accepted:
                accepted.pop(key); rejected.append(key)
    return accepted, rejected


def validate_payload(data):
    """Validate shape before grounding; do not silently coerce malformed clinical entities."""
    if not isinstance(data, dict):
        raise ValueError('SOAP must be an object')
    for key in ('diagnoses', 'medications', 'procedures'):
        values = data.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            raise ValueError(f'{key} must be a list')
        for item in values:
            if not isinstance(item, dict) or not isinstance(item.get('term'), str) or not item['term'].strip():
                raise ValueError(f'{key} item requires a nonempty term')
    if data.get('vitals') is not None and not isinstance(data['vitals'], dict):
        raise ValueError('vitals must be an object')
    if data.get('speakers') is not None and not isinstance(data['speakers'], dict):
        raise ValueError('speakers must be an object')
    for key in ('chief_complaint', 'history_of_present_illness', 'past_medical_history',
                'allergies', 'physical_exam', 'patient_education', 'follow_up_plan'):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f'{key} must be a string or null')
