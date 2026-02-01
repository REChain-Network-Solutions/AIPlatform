<?php

namespace App\Extensions\PhotoStudio\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class PhotoStudioRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'action'      => ['string', 'max:255'],
            'photo'       => $this->photoRule(),
            'description' => [$this->descriptionRule(), 'nullable', 'string', 'max:255'],
        ];
    }

    public function photoRule(): array
    {
        if ($this->request->get('action') === 'text_to_image') {
            return ['sometimes', 'nullable'];
        }

        return ['required', 'image', 'mimes:jpeg,png,jpg,gif,svg', 'max:5120'];
    }

    public function descriptionRule(): string
    {
        if ($this->request->get('action') === 'replace_background') {
            return 'required';
        }

        if ($this->request->get('action') === 'sketch_to_image') {
            return 'required';
        }

        if ($this->request->get('action') === 'text_to_image') {
            return 'required';
        }

        return 'sometimes';
    }
}
