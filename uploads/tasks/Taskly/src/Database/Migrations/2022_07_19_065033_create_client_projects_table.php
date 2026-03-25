<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateClientProjectsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if(!Schema::hasTable('client_projects'))
        {
            Schema::create('client_projects', function (Blueprint $table) {
                $table->id();
                $table->integer('client_id');
                $table->integer('project_id');
                $table->integer('is_active')->default(1);
                $table->text('permission')->nullable();
                $table->timestamps();
            });
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('client_projects');
    }
}
