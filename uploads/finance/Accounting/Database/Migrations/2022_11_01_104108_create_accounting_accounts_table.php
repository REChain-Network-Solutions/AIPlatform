<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateAccountingAccountsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('accounting_accounts', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('name');
            $table->string('gl_code')->nullable();
            $table->integer('business_id');
            $table->string('account_primary_type')->nullable();
            $table->bigInteger('account_sub_type_id')->nullable();
            $table->bigInteger('detail_type_id')->nullable();
            $table->bigInteger('parent_account_id')->nullable();
            // $table->decimal('balance')->default(0);
            // $table->date('balance_as_of')->nullable();
            $table->longText('description')->nullable();
            $table->string('status')->nullable();
            $table->integer('created_by');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('accounting_accounts');
    }
}
