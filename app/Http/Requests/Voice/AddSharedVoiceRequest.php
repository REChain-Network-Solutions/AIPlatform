<?php

namespace App\Http\Requests\Voice;

use Illuminate\Foundation\Http\FormRequest;

class AddSharedVoiceRequest extends FormRequest
{
    public function authorize(): bool
    {
        return $this->user() !== null;
    }

    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            'public_user_id' => 'required|string|max:255',
            'voice_id'       => 'required|string|max:255',
            'name'           => 'required|string|max:255',
            'preview_url'    => 'nullable|string|max:2048',
            'language'       => 'nullable|string|max:16',
        ];
    }
}
