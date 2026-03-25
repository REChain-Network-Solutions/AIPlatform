<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Comment extends Model
{
    use HasFactory;


    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\CommentFactory::new();
    }

    protected $fillable = [
        'comment','created_by','task_id','user_type'
    ];
    public function user(){
        return $this->hasOne('App\Models\User','id','created_by');
    }

}
