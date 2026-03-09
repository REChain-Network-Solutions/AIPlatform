<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->index('team_id', 'idx_users_team_id');
            $table->index('team_manager_id', 'idx_users_team_manager_id');
            $table->index('type', 'idx_users_type');
            $table->index('last_activity_at', 'idx_users_last_activity_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->dropIndex('idx_users_team_id');
            $table->dropIndex('idx_users_team_manager_id');
            $table->dropIndex('idx_users_type');
            $table->dropIndex('idx_users_last_activity_at');
        });
    }
};
