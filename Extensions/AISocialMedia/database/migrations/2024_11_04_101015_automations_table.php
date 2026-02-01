<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('automations')) {
            return;
        }

        Schema::create('automations', function (Blueprint $table) {
            $table->id();
            $table->longText('custom')->nullable();
            $table->longText('value')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void {}
};
