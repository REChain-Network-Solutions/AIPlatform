<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class BugFile extends Model
{
    use HasFactory;

    protected $fillable = [
        'file','name','extension','file_size','created_by','bug_id','user_type'
    ];

    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\BugFileFactory::new();
    }
}
