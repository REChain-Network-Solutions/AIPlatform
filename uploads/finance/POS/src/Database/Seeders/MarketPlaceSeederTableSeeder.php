<?php

namespace Workdo\Pos\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Database\Eloquent\Model;
use Workdo\LandingPage\Entities\MarketplacePageSetting;


class MarketPlaceSeederTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        Model::unguard();
        $module = 'Pos';

        $data['product_main_banner'] = '';
        $data['product_main_status'] = 'on';
        $data['product_main_heading'] = 'POS';
        $data['product_main_description'] = '<p>Welcome to the POS Module in Dash SaaS, your ultimate solution for streamlined sales management. Our platform offers a user-friendly dashboard where you can easily navigate through your sales activities. At the heart of the dashboard is a dynamic chart representing your sales reports, providing valuable insights into your business is performance. Whether you are a small retail store or a bustling restaurant, our POS Module empowers you to efficiently manage your sales process from start to finish.</p>';
        $data['product_main_demo_link'] = '#';
        $data['product_main_demo_button_text'] = 'View Live Demo';
        $data['dedicated_theme_heading'] = '<h2>Dash SaaS POS: Effortless Sales Management</h2>';
        $data['dedicated_theme_description'] = '<p>Simplify sales management with Dash SaaS POS. Track revenue, manage products, process orders, and utilize barcode functionality for seamless transactions.</p>';
        $data['dedicated_theme_sections'] = '[
                                                {
                                                    "dedicated_theme_section_image": "",
                                                    "dedicated_theme_section_heading": "Real-Time Revenue Tracking",
                                                    "dedicated_theme_section_description": "<p>With the POS Module, tracking your revenue in real time has never been easier. At the top of the dashboard, you will find a dedicated section displaying the total sales amount. This feature allows you to monitor your revenue at a glance, empowering you to make informed decisions about your business is financial health. Whether you are analyzing daily, weekly, or monthly sales trends, our platform provides the tools you need to stay on top of your finances effortlessly.<\/p>",
                                                    "dedicated_theme_section_cards": {
                                                    "1": {
                                                        "title": null,
                                                        "description": null
                                                    }
                                                    }
                                                },
                                                {
                                                    "dedicated_theme_section_image": "",
                                                    "dedicated_theme_section_heading": "Effortless Product Management",
                                                    "dedicated_theme_section_description": "<p>Managing your products for sale is a breeze with the POS Module in Dash SaaS. Our platform offers robust product management features, allowing you to add, edit, or remove products with ease. Whether you are introducing new items to your inventory or updating existing ones, our intuitive interface makes the process seamless. With our product management tools, you can ensure that your inventory is always up-to-date, helping you serve your customers more effectively.<\/p>",
                                                    "dedicated_theme_section_cards": {
                                                    "1": {
                                                        "title": null,
                                                        "description": null
                                                    }
                                                    }
                                                },
                                                {
                                                    "dedicated_theme_section_image": "",
                                                    "dedicated_theme_section_heading": "Efficient Order Processing",
                                                    "dedicated_theme_section_description": "<p>Streamline your order processing with the POS Module is efficient order management system. Our platform allows you to process orders quickly and accurately, ensuring a seamless experience for both you and your customers. From customizing orders based on customer preferences to tracking order status in real time, our POS Module provides the flexibility and functionality you need to keep your business running smoothly. With our order processing features, you can focus on delivering exceptional service to your customers, every time.<\/p>",
                                                    "dedicated_theme_section_cards": {
                                                    "1": {
                                                        "title": null,
                                                        "description": null
                                                    }
                                                    }
                                                },
                                                {
                                                    "dedicated_theme_section_image": "",
                                                    "dedicated_theme_section_heading": "Streamlined Barcode Printing",
                                                    "dedicated_theme_section_description": "<p>Enhance efficiency in your sales workflow with Dash SaaS POS is streamlined barcode printing feature. Easily generate barcodes for products and orders by selecting items and specifying quantities. This functionality simplifies inventory management and accelerates checkout processes. Improve operational productivity by swiftly printing as many barcodes as needed, empowering your business to meet customer demands effectively. Experience convenience and flexibility with Dash SaaS POS is barcode printing capability, tailored to optimize your sales operations.<\/p>",
                                                    "dedicated_theme_section_cards": {
                                                    "1": {
                                                        "title": null,
                                                        "description": null
                                                    }
                                                    }
                                                }
                                            ]';
        $data['dedicated_theme_sections_heading'] = '';
        $data['screenshots'] = '[{"screenshots":"","screenshots_heading":"POS"},{"screenshots":"","screenshots_heading":"POS"},{"screenshots":"","screenshots_heading":"POS"},{"screenshots":"","screenshots_heading":"POS"},{"screenshots":"","screenshots_heading":"POS"},{"screenshots":"","screenshots_heading":"POS"}]';
        $data['addon_heading'] = '<h2>Why choose dedicated modules<b> for Your Business?</b></h2>';
        $data['addon_description'] = '<p>With Dash, you can conveniently manage all your business functions from a single location.</p>';
        $data['addon_section_status'] = 'on';
        $data['whychoose_heading'] = 'Why choose dedicated modulesfor Your Business?';
        $data['whychoose_description'] = '<p>With Dash, you can conveniently manage all your business functions from a single location.</p>';
        $data['pricing_plan_heading'] = 'Empower Your Workforce with DASH';
        $data['pricing_plan_description'] = '<p>Access over Premium Add-ons for Accounting, HR, Payments, Leads, Communication, Management, and more, all in one place!</p>';
        $data['pricing_plan_demo_link'] = '#';
        $data['pricing_plan_demo_button_text'] = 'View Live Demo';
        $data['pricing_plan_text'] = '{"1":{"title":"Pay-as-you-go"},"2":{"title":"Unlimited installation"},"3":{"title":"Secure cloud storage"}}';
        $data['whychoose_sections_status'] = 'on';
        $data['dedicated_theme_section_status'] = 'on';

        foreach($data as $key => $value){
            if(!MarketplacePageSetting::where('name', '=', $key)->where('module', '=', 'Pos')->exists()){
                MarketplacePageSetting::updateOrCreate(
                [
                    'name' => $key,
                    'module' => 'Pos'

                ],
                [
                    'value' => $value
                ]);
            }
        }
    }
}
