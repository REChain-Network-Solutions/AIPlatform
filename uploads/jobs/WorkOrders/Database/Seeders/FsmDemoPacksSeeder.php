<?php
namespace Modules\WorkOrders\Database\Seeders; use Illuminate\Database\Seeder; use Illuminate\Support\Facades\DB;
class FsmDemoPacksSeeder extends Seeder{
  public function run(): void {
    $verticals=['general_trades','hvac'];
    foreach($verticals as $v){
      $clientId = DB::table('clients')->insertGetId(['name'=>ucfirst($v).' Demo Client','email'=>$v.'@example.com','created_at'=>now(),'updated_at'=>now()]);
      $woId = DB::table('work_orders')->insertGetId(['client_id'=>$clientId,'client_name'=>ucfirst($v).' Demo Client','status'=>'pending','scheduled_at'=>now()->addDays(1),'created_at'=>now(),'updated_at'=>now()]);
      if (DB::table('users')->where('id',1)->exists()){
        DB::table('work_order_contractor_assignments')->insert(['work_order_id'=>$woId,'user_id'=>1,'scheduled_at'=>now()->addDays(1)->setTime(9,0),'duration_minutes'=>90,'created_at'=>now(),'updated_at'=>now()]);
      }
    }
  }
}