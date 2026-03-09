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
        Schema::table('menus', function (Blueprint $table) {
            $table->index('parent_id', 'idx_menus_parent_id');
            $table->index(['parent_id', 'order'], 'idx_menus_parent_id_order');
            $table->index('is_active', 'idx_menus_is_active');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('menus', function (Blueprint $table) {
            $table->dropIndex('idx_menus_parent_id');
            $table->dropIndex('idx_menus_parent_id_order');
            $table->dropIndex('idx_menus_is_active');
        });
    }
};
