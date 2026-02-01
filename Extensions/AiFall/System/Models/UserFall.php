<?php

namespace App\Extensions\AiFall\System\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class UserFall extends Model
{
    use HasFactory;

    protected $table = 'user_fall';

    protected $fillable = [
        'user_id',
        'prompt',
        'prompt_image_url',
        'status',
        'request_id',
        'response_url',
        'model',
        'video_url',
    ];
}
