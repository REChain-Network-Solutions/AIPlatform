<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Referer extends Model
{
    use HasFactory;

    protected $fillable = ['session_id', 'referer', 'domain', 'created_at'];

    public $timestamps = false;
}
