<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('custom_module_install_logs')) {
            return;
        }

        Schema::create('custom_module_install_logs', function (Blueprint $table) {
            $table->id();
            $table->string('file_name')->nullable();
            $table->json('payload')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('custom_module_install_logs');
    }
};
