<?php

namespace App\Extensions\AIRealtimeImage\System\Http\Requests;

use App\Domains\Entity\Enums\EntityEnum;
use App\Models\SettingTwo;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Support\Facades\Auth;

class AIRealtimeImageRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'user_id'         => 'required|integer',
            'prompt'          => 'required',
            'style'           => 'nullable|string',
            'model'           => 'required',
            'payload'         => 'array',
            'status'          => 'required',
            'disk'            => 'required',
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'user_id' => Auth::id(),
            'model'   => EntityEnum::BLACK_FOREST_LABS_FLUX_1_SCHNELL->slug(),
            'status'  => 'pending',
            'disk'    => SettingTwo::getCache()?->getAttribute('ai_image_storage'),
            'payload' => [
                'prompt' => $this->input('prompt'),
                'model'  => EntityEnum::BLACK_FOREST_LABS_FLUX_1_SCHNELL->value,
                'width'  => 1024,
                'height' => 768,
                'seed'   => 123,
                'steps'  => 3,
            ],
        ]);
    }
}
