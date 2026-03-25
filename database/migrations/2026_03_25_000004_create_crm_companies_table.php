<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_companies', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->string('name');
            $table->string('domain')->nullable();
            $table->string('website')->nullable();
            $table->string('phone')->nullable();
            $table->string('industry')->nullable();
            $table->string('type')->nullable(); // prospect, customer, partner, vendor
            $table->integer('employee_count')->nullable();
            $table->decimal('annual_revenue', 15, 2)->nullable();
            $table->text('address')->nullable();
            $table->string('city')->nullable();
            $table->string('state')->nullable();
            $table->string('country', 2)->nullable();
            $table->string('logo_path')->nullable();
            $table->text('description')->nullable();
            $table->json('custom_fields')->nullable();
            $table->timestamps();
            $table->softDeletes();

            $table->index('user_id');
            $table->index('name');
        });

        // Now add FK to crm_contacts now that crm_companies exists
        Schema::table('crm_contacts', function (Blueprint $table) {
            $table->foreign('crm_company_id')->references('id')->on('crm_companies')->nullOnDelete();
        });
    }

    public function down(): void
    {
        Schema::table('crm_contacts', function (Blueprint $table) {
            $table->dropForeign(['crm_company_id']);
        });

        Schema::dropIfExists('crm_companies');
    }
};
