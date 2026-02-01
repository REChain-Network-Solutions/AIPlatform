<?php

namespace App\Extensions\AISocialMedia\System\Models;

use App\Models\User;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class LinkedinTokens extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'access_token',
    ];

    public function user()
    {
        return $this->belongsTo(User::class);
    }
}
