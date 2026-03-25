<?php

namespace Modules\Quotes\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreQuoteRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'client_id' => ['nullable','integer'],
            'currency' => ['required','string','max:8'],
            'valid_until' => ['nullable','date'],
            'notes' => ['nullable','string'],
            'items' => ['required','array','min:1'],
            'items.*.description' => ['required','string'],
            'items.*.qty' => ['required','numeric','min:0.01'],
            'items.*.unit_price' => ['required','numeric','min:0'],
            'items.*.tax_rate' => ['nullable','numeric','min:0','max:1'],
        ];
    }
}
