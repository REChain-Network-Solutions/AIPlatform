<?php

use App\Domains\Entity\Models\Entity;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    public function up(): void
    {
        Entity::query()
            ->where('engine', 'anthropic')
            ->whereIn('key', [
                'claude-opus-5',
            ])
            ->update(['image' => 'upload/enginelogo/claude_logo.svg']);
    }

    public function down(): void
    {
        Entity::query()
            ->where('engine', 'anthropic')
            ->whereIn('key', [
                'claude-opus-5',
            ])
            ->where('image', 'upload/enginelogo/claude_logo.svg')
            ->update(['image' => null]);
    }
};
