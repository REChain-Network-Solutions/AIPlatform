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
        // Create module installation logs table
        Schema::create('module_installation_logs', function (Blueprint $table) {
            $table->id();
            $table->string('module_name')->index();
            $table->string('event_type')->index();
            $table->enum('level', ['debug', 'info', 'warning', 'error'])->index();
            $table->json('data')->nullable();
            $table->foreignId('user_id')->nullable()->constrained('users')->nullOnDelete();
            $table->timestamp('logged_at')->index();
            $table->timestamps();
        });

        // Create module configuration table
        Schema::create('module_configurations', function (Blueprint $table) {
            $table->id();
            $table->string('key');
            $table->longText('value')->nullable();
            $table->string('environment')->default('global');
            $table->timestamps();
            $table->unique(['key', 'environment']);
            $table->index(['environment']);
        });

        // Create module dependencies table (optional but recommended)
        Schema::create('module_dependencies', function (Blueprint $table) {
            $table->id();
            $table->string('module_name');
            $table->string('depends_on'); // Dependency module name
            $table->string('required_version')->default('*');
            $table->timestamps();
            $table->unique(['module_name', 'depends_on']);
        });

        // Create module test results table
        Schema::create('module_test_results', function (Blueprint $table) {
            $table->id();
            $table->foreignId('install_id')->constrained('module_install_logs')->cascadeOnDelete();
            $table->string('module_name');
            $table->string('test_category'); // pre_install, post_install, compatibility, performance, security
            $table->string('test_name');
            $table->boolean('success');
            $table->text('message')->nullable();
            $table->json('details')->nullable();
            $table->timestamps();
            $table->index(['module_name', 'test_category']);
            $table->index(['install_id']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('module_test_results');
        Schema::dropIfExists('module_dependencies');
        Schema::dropIfExists('module_configurations');
        Schema::dropIfExists('module_installation_logs');
    }
};
