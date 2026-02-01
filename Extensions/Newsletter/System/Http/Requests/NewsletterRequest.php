<?php

namespace App\Extensions\Newsletter\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class NewsletterRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'title'   => 'required',
            'subject' => 'required',
            'content' => 'required',
            'system'  => 'required|boolean',
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'system'  => false,
            'content' => $this->get('content_ace'),
        ]);
    }
}
