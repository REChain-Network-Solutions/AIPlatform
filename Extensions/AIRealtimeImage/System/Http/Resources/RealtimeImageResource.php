<?php

declare(strict_types=1);

namespace App\Extensions\AIRealtimeImage\System\Http\Resources;

use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use JsonSerializable;

class RealtimeImageResource extends JsonResource
{
    public function toArray(Request $request): array|Arrayable|JsonSerializable
    {
        return [
            'id'         => $this->id,
            'prompt'     => $this->prompt,
            'style'      => $this->style,
            'model'      => $this->model,
            'status'     => $this->status,
            'image'      => $this->image,
            'image_url'  => $this->image_url,
            'created_at' => $this->created_at,
            'updated_at' => $this->updated_at,
        ];
    }
}
