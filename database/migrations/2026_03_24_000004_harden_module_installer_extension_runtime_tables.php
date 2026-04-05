<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('module_dependencies')) {
            Schema::table('module_dependencies', function (Blueprint $table) {
                if (!Schema::hasColumn('module_dependencies', 'status')) {
                    $table->string('status')->default('pending')->after('required_version')->index();
                }
                if (!Schema::hasColumn('module_dependencies', 'source')) {
                    $table->string('source')->nullable()->after('status');
                }
                if (!Schema::hasColumn('module_dependencies', 'notes')) {
                    $table->text('notes')->nullable()->after('source');
                }
            });
        }

        if (Schema::hasTable('module_test_results')) {
            Schema::table('module_test_results', function (Blueprint $table) {
                if (!Schema::hasColumn('module_test_results', 'details')) {
                    $table->json('details')->nullable()->after('message');
                }
            });
        }

        if (Schema::hasTable('module_installation_logs')) {
            Schema::table('module_installation_logs', function (Blueprint $table) {
                if (!Schema::hasColumn('module_installation_logs', 'install_id')) {
                    $table->unsignedBigInteger('install_id')->nullable()->after('user_id')->index();
                }
            });
        }
    }

    public function down(): void
    {
        if (Schema::hasTable('module_dependencies')) {
            Schema::table('module_dependencies', function (Blueprint $table) {
                foreach (['notes', 'source', 'status'] as $column) {
                    if (Schema::hasColumn('module_dependencies', $column)) {
                        $table->dropColumn($column);
                    }
                }
            });
        }

        if (Schema::hasTable('module_test_results') && Schema::hasColumn('module_test_results', 'details')) {
            Schema::table('module_test_results', function (Blueprint $table) {
                $table->dropColumn('details');
            });
        }

        if (Schema::hasTable('module_installation_logs') && Schema::hasColumn('module_installation_logs', 'install_id')) {
            Schema::table('module_installation_logs', function (Blueprint $table) {
                $table->dropColumn('install_id');
            });
        }
    }
};
