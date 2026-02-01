<?php

namespace App\Extensions\PhotoStudio\System\Models;

use Illuminate\Database\Eloquent\Model;

class PhotoStudio extends Model
{
    protected $fillable = [
        'user_id',
        'photo',
        'payload',
        'credits',
        'status',
        'request_id',
    ];
}
