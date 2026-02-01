<?php

namespace App\Extensions\AiAvatar\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class AiAvatarRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'avatar'          => 'required',
            'title'           => 'required',
            'description'     => 'required',
            'style'           => 'required',
            'scriptText'      => 'required',
            'backgroundColor' => 'required',
            'background'      => 'required',
            'horizontalAlign' => ['sometimes', 'nullable'],
        ];
    }
}
