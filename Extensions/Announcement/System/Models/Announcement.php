<?php

namespace App\Extensions\Announcement\System\Models;

use App\Extensions\Announcement\System\Enum\AnnouncementType;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Announcement extends Model
{
    use HasFactory;

    protected $fillable = ['title', 'type', 'active'];

    protected $casts = [
        'active' => 'boolean',
        'type' 	 => AnnouncementType::class,
    ];
}
