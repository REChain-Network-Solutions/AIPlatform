<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleTestResult extends Model
{
    protected $table = 'module_test_results';

    protected $fillable = [
        'install_id',
        'module_name',
        'test_category',
        'test_name',
        'success',
        'message',
        'details',
    ];

    protected $casts = [
        'success' => 'boolean',
        'details' => 'array',
    ];
}
