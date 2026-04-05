<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleDependency extends Model
{
    protected $table = 'module_dependencies';

    protected $fillable = [
        'module_name',
        'depends_on',
        'required_version',
        'status',
        'source',
        'notes',
    ];
}
