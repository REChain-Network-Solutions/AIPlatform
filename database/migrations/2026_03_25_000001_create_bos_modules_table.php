<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('bos_modules', function (Blueprint $table) {
            $table->id();
            $table->string('key')->unique();             // e.g. 'crm', 'hr', 'projects', 'finance'
            $table->string('name');                      // e.g. 'CRM'
            $table->string('description')->nullable();
            $table->string('icon')->default('tabler-layout-grid'); // Tabler icon name
            $table->string('color')->default('#6366f1'); // accent color
            $table->string('route_prefix')->nullable();  // e.g. 'crm'
            $table->boolean('is_active')->default(true);
            $table->boolean('is_core')->default(false);  // core = cannot be disabled
            $table->integer('order')->default(0);
            $table->json('settings')->nullable();         // module-level config
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('bos_modules');
    }
};
