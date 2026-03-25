<?php

namespace Modules\EInvoice\Helper\XmlGenerator;

use App\Models\Invoice;
use App\Models\Tax;
use Modules\EInvoice\Entities\EInvoiceCompanySetting;
use Saloon\XmlWrangler\Data\Element;

class Romania
{

    public static function generate(Invoice $invoice)
    {
        $array = [
            'cbc:CustomizationID' => 'urn:cen.eu:en16931:2017#compliant#urn:fdc:peppol.eu:2017:poacc:billing:3.0',
            'cbc:ProfileID' => 'urn:fdc:peppol.eu:2017:poacc:billing:01:1.0',
            'cbc:ID' => $invoice->invoice_number,
            'cbc:IssueDate' => $invoice->issue_date->format('Y-m-d'),
            'cbc:DueDate' => $invoice->due_date->format('Y-m-d'),
            'cbc:InvoiceTypeCode' => 380,
            'cbc:Note' => $invoice->note,
            'cbc:TaxPointDate' => $invoice->due_date->format('Y-m-d'),
            'cbc:DocumentCurrencyCode' => $invoice->currency?->currency_code,
            'cbc:TaxCurrencyCode' => $invoice->currency?->currency_code,
            'cbc:BuyerReference' => $invoice->invoice_number,
            'cac:AccountingSupplierParty' => self::accountingSupplierParty($invoice),
            'cac:AccountingCustomerParty' => self::accountingCustomerParty($invoice),
            'cac:TaxTotal' => self::taxTotal($invoice),
            'cac:LegalMonetaryTotal' => [
                'cbc:LineExtensionAmount' => Element::make(($invoice->sub_total - $invoice->discount), attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cbc:TaxExclusiveAmount' => Element::make(($invoice->sub_total - $invoice->discount), attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cbc:TaxInclusiveAmount' => Element::make($invoice->total, attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cbc:PayableAmount' => Element::make($invoice->total, attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
            ],
            'cac:InvoiceLine' => [
                'cbc:ID' => $invoice->id,
                'cbc:Note' => $invoice->note,
                'cbc:InvoicedQuantity' => Element::make($invoice->items->count(), attributes: [
                    'unitCode' => 'C62',
                ]),
                'cbc:LineExtensionAmount' => Element::make(($invoice->sub_total - $invoice->discount), attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cbc:AccountingCost' => $invoice->accounting_cost,
                'cac:Item' => self::invoiceItems($invoice),
                'cac:Price' => [
                    'cbc:PriceAmount' => Element::make(($invoice->sub_total - $invoice->discount), attributes: [
                        'currencyID' => $invoice->currency?->currency_code,
                    ]),
                ],
            ],
        ];

        // remove empty elements
        $array = self::removeEmptyElements($array);

        return $array;
    }

    public static function accountingSupplierParty(Invoice $invoice)
    {
        $eInvoiceSettings = EInvoiceCompanySetting::first();
        $address = $invoice->address;

        $partyEndpointID = '';
        $partyCompanyID = '';

        if ($eInvoiceSettings) {
            $partyEndpointID = Element::make($eInvoiceSettings->electronic_address, attributes: [
                'schemeID' => $eInvoiceSettings->electronic_address_scheme,
            ]);

            $partyCompanyID = Element::make($eInvoiceSettings->e_invoice_company_id, attributes: [
                'schemeID' => $eInvoiceSettings->e_invoice_company_id_scheme,
            ]);
        }

        return [
            'cac:Party' => [
                'cbc:EndpointID' => $partyEndpointID,
                'cac:PartyName' => [
                    'cbc:Name' => $invoice->company->company_name,
                ],
                'cac:PostalAddress' => [
                    'cbc:StreetName' => $address->address,
                    'cbc:CityName' => $address->location,
                    'cac:Country' => [
                        'cbc:IdentificationCode' => $address->country?->iso,
                    ],
                ],
                'cac:PartyTaxScheme' => [
                    'cbc:CompanyID' => $address->tax_name.$address->tax_number,
                    'cac:TaxScheme' => [
                        'cbc:ID' => $address->tax_name,
                    ],
                ],
                'cac:PartyLegalEntity' => [
                    'cbc:RegistrationName' => $invoice->company->company_name,
                    'cbc:CompanyID' => $partyCompanyID,
                ],
                'cac:Contact' => [
                    'cbc:Name' => $invoice->company->company_name,
                    'cbc:Telephone' => $invoice->company->company_phone,
                    'cbc:ElectronicMail' => $invoice->company->company_email,
                ],
            ],
        ];
    }

    public static function accountingCustomerParty(Invoice $invoice)
    {
        $address = $invoice->address;
        $client = $invoice->client;
        $clientDetails = $client->clientDetails;

        return [
            'cac:Party' => [
                'cbc:EndpointID' => Element::make($clientDetails->electronic_address, attributes: [
                    'schemeID' => $clientDetails->electronic_address_scheme,
                ]),
                'cac:PostalAddress' => [
                    'cbc:StreetName' => $clientDetails?->address,
                    'cbc:CityName' => $clientDetails?->city,
                    'cbc:PostalZone' => $clientDetails?->postal_code,
                    'cbc:CountrySubentity' => $clientDetails?->state,
                    'cac:Country' => [
                        'cbc:IdentificationCode' => $client?->country?->iso,
                    ],
                ],
                'cac:PartyTaxScheme' => [
                    'cbc:CompanyID' => $address->tax_name.$clientDetails->gst_number,
                    'cac:TaxScheme' => [
                        'cbc:ID' => $address->tax_name,
                    ],
                ],
                'cac:PartyLegalEntity' => [
                    'cbc:RegistrationName' => $clientDetails?->company_name,
                ],
                'cac:Contact' => [
                    'cbc:Name' => $client?->name,
                    'cbc:Telephone' => $client?->mobile,
                    'cbc:ElectronicMail' => $client?->email,
                ],
            ],
        ];
    }

    public static function invoiceItems(Invoice $invoice)
    {
        $items = [];

        foreach ($invoice->items as $item) {

            $taxPercent = 0;
            $tax = null;

            if ($item->taxes) {
                $taxesIds = json_decode($item->taxes);
                $taxes = Tax::whereIn('id', $taxesIds)->withTrashed()->get();
                $taxPercent = $taxes->avg('rate_percent');
                $tax = $taxes->first();
            }

            $items[] = [
                'cbc:Description' => $item->item_summary,
                'cbc:Name' => $item->item_name,
                'cac:ClassifiedTaxCategory' => [
                    'cbc:ID' => 'S',
                    'cbc:Percent' => $taxPercent,
                    'cac:TaxScheme' => [
                        'cbc:ID' => $tax?->tax_name,
                    ],
                ],
            ];
        }

        return $items;
    }

    public static function taxTotal(Invoice $invoice)
    {
        $taxableAmount = $invoice->sub_total - $invoice->discount;
        $taxAmount = $invoice->total - $taxableAmount;
        $percent = ($taxAmount / $taxableAmount) * 100;

        return [
            'cbc:TaxAmount' => Element::make($taxAmount, attributes: [
                'currencyID' => $invoice->currency?->currency_code,
            ]),
            'cac:TaxSubtotal' => [
                'cbc:TaxableAmount' => Element::make($taxableAmount, attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cbc:TaxAmount' => Element::make($taxAmount, attributes: [
                    'currencyID' => $invoice->currency?->currency_code,
                ]),
                'cac:TaxCategory' => [
                    'cbc:ID' => 'S',
                    'cbc:Percent' => $percent,
                    'cac:TaxScheme' => [
                        'cbc:ID' => $invoice->address?->tax_name,
                    ],
                ],
            ],
        ];

    }

    public static function removeEmptyElements($array)
    {
        foreach ($array as $key => $value) {
            if (is_array($value)) {
                $array[$key] = self::removeEmptyElements($value);
            }
            if (empty($array[$key])) {
                unset($array[$key]);
            }
        }
        return $array;
    }

}
