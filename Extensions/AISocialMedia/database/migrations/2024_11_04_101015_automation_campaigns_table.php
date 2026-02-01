<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('automation_campaigns')) {
            return;
        }

        Schema::create('automation_campaigns', function (Blueprint $table) {
            $table->id();
            $table->integer('user_id');
            $table->string('name')->nullable();
            $table->text('target_audience')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void {}
};
