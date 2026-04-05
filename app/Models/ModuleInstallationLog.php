<?php

namespace App\Models;

class ModuleInstallationLog extends BaseModel
{
    protected $table = 'module_installation_logs';

    protected $guarded = ['id'];

    protected $casts = [
        'data' => 'array',
        'logged_at' => 'datetime',
    ];
}
