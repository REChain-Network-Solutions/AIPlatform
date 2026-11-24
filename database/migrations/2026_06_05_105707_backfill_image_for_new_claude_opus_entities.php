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
                'claude-opus-4-8',
                'claude-opus-4-7',
            ])
            ->whereNull('image')
            ->update(['image' => 'upload/enginelogo/claude_logo.svg']);
    }

    public function down(): void
    {
        Entity::query()
            ->where('engine', 'anthropic')
            ->whereIn('key', [
                'claude-opus-4-8',
                'claude-opus-4-7',
            ])
            ->where('image', 'upload/enginelogo/claude_logo.svg')
            ->update(['image' => null]);
    }
};
