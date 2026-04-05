<?php

use App\Models\Company;
use App\Observers\CompanyObserver;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{

    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Company::get()->each(function ($company) {
            (new CompanyObserver())->createModuleSettings($company);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        //
    }

};
