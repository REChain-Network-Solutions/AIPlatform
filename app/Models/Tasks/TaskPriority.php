<?php

namespace App\Models\Tasks;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class TaskPriority extends Model
{
    protected $table = 'task_priorities';

    protected $fillable = ['user_id', 'name', 'color', 'level', 'is_default'];

    protected $casts = [
        'is_default' => 'boolean',
        'level'      => 'integer',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function tasks(): HasMany
    {
        return $this->hasMany(Task::class, 'task_priority_id');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId)->orderBy('level');
    }
}
