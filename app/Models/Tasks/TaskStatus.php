<?php

namespace App\Models\Tasks;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class TaskStatus extends Model
{
    protected $table = 'task_statuses';

    protected $fillable = ['user_id', 'name', 'color', 'position', 'is_default', 'is_completed_status'];

    protected $casts = [
        'is_default'          => 'boolean',
        'is_completed_status' => 'boolean',
        'position'            => 'integer',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function tasks(): HasMany
    {
        return $this->hasMany(Task::class, 'task_status_id');
    }

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId)->orderBy('position');
    }
}
