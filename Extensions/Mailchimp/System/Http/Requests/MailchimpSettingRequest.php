<?php

namespace App\Extensions\Mailchimp\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class MailchimpSettingRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'mailchimp_api_key'  => 'required',
            'mailchimp_list_id'  => 'required',
            'mailchimp_register' => 'sometimes',
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'mailchimp_register' => $this->has('mailchimp_register'),
        ]);
    }
}
