<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class ClientProject extends Model
{
    use HasFactory;

    protected $fillable = [
        'client_id','project_id','is_active','permission'
    ];

    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\ClientProjectFactory::new();
    }
    public function getByProjects($project_id){
        $check = self::where('project_id',$project_id)->pluck('client_id');
        if($check->count() > 0){
            return $check->toArray();
        }else{
            return [];
        }
    }
    public function project()
    {
        return  $this->hasOne(Project::class,'id','project_id');
    }
}
