<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Request\Receipts;

use Exception;
use InvalidArgumentException;
use stdClass;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Helpers\ProductCode;
use YooKassa\Helpers\Random;
use YooKassa\Helpers\StringObject;
use YooKassa\Model\Metadata;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Request\Receipts\ReceiptResponseItem;

/**
 * ReceiptResponseItemTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptResponseItemTest extends AbstractTestCase
{
    protected ReceiptResponseItem $object;

    /**
     * @return ReceiptResponseItem
     */
    protected function getTestInstance(): ReceiptResponseItem
    {
        return new ReceiptResponseItem();
    }

    /**
     * @return void
     */
    public function testReceiptResponseItemClassExists(): void
    {
        $this->object = $this->getMockBuilder(ReceiptResponseItem::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ReceiptResponseItem::class));
        $this->assertInstanceOf(ReceiptResponseItem::class, $this->object);
    }

    /**
     * Test property "description"
     * @dataProvider validDescriptionDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDescription(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setDescription($value);
        self::assertNotNull($instance->getDescription());
        self::assertNotNull($instance->description);
        self::assertEquals($value, is_array($value) ? $instance->getDescription()->toArray() : $instance->getDescription());
        self::assertEquals($value, is_array($value) ? $instance->description->toArray() : $instance->description);
    }

    /**
     * Test invalid property "description"
     * @dataProvider invalidDescriptionDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDescription(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDescription($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
    }

    /**
     * Test property "amount"
     * @dataProvider validAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPrice($value);
        self::assertNotNull($instance->getPrice());
        self::assertNotNull($instance->amount);
        self::assertEquals($value, is_array($value) ? $instance->getPrice()->toArray() : $instance->getPrice());
        self::assertEquals($value['value'], $instance->getPrice()->getValue());
        self::assertEquals($value['currency'], $instance->getPrice()->getCurrency());
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }

    /**
     * Test property "vat_code"
     * @dataProvider validVatCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testVatCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setVatCode($value);
        self::assertNotNull($instance->getVatCode());
        self::assertNotNull($instance->vat_code);
        self::assertEquals($value, is_array($value) ? $instance->getVatCode()->toArray() : $instance->getVatCode());
        self::assertEquals($value, is_array($value) ? $instance->vat_code->toArray() : $instance->vat_code);
    }

    /**
     * Test invalid property "vat_code"
     * @dataProvider invalidVatCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidVatCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setVatCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validVatCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_vat_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidVatCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_vat_code'));
    }

    /**
     * Test property "quantity"
     * @dataProvider validQuantityDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setQuantity($value);
        self::assertNotNull($instance->getQuantity());
        self::assertNotNull($instance->quantity);
        self::assertEquals($value, is_array($value) ? $instance->getQuantity()->toArray() : $instance->getQuantity());
        self::assertEquals($value, is_array($value) ? $instance->quantity->toArray() : $instance->quantity);
    }

    /**
     * Test invalid property "quantity"
     * @dataProvider invalidQuantityDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setQuantity($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
     * Test property "measure"
     * @dataProvider validMeasureDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMeasure(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMeasure());
        self::assertEmpty($instance->measure);
        $instance->setMeasure($value);
        self::assertEquals($value, is_array($value) ? $instance->getMeasure()->toArray() : $instance->getMeasure());
        self::assertEquals($value, is_array($value) ? $instance->measure->toArray() : $instance->measure);
        if (!empty($value)) {
            self::assertNotNull($instance->getMeasure());
            self::assertNotNull($instance->measure);
        }
    }

    /**
     * Test invalid property "measure"
     * @dataProvider invalidMeasureDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMeasure(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMeasure($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMeasureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_measure'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMeasureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_measure'));
    }

    /**
     * Test property "mark_quantity"
     * @dataProvider validMarkQuantityDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMarkQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMarkQuantity());
        self::assertEmpty($instance->mark_quantity);
        $instance->setMarkQuantity($value);
        self::assertEquals($value, is_array($value) ? $instance->getMarkQuantity()->toArray() : $instance->getMarkQuantity());
        self::assertEquals($value, is_array($value) ? $instance->mark_quantity->toArray() : $instance->mark_quantity);
        if (!empty($value)) {
            self::assertNotNull($instance->getMarkQuantity());
            self::assertNotNull($instance->mark_quantity);
        }
    }

    /**
     * Test invalid property "mark_quantity"
     * @dataProvider invalidMarkQuantityDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMarkQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMarkQuantity($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMarkQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_quantity'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMarkQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_quantity'));
    }

    /**
     * Test property "payment_subject"
     * @dataProvider validPaymentSubjectDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentSubject(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentSubject());
        self::assertEmpty($instance->payment_subject);
        $instance->setPaymentSubject($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentSubject()->toArray() : $instance->getPaymentSubject());
        self::assertEquals($value, is_array($value) ? $instance->payment_subject->toArray() : $instance->payment_subject);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentSubject());
            self::assertNotNull($instance->payment_subject);
        }
    }

    /**
     * Test invalid property "payment_subject"
     * @dataProvider invalidPaymentSubjectDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentSubject(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentSubject($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentSubjectDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_subject'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentSubjectDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_subject'));
    }

    /**
     * Test property "payment_mode"
     * @dataProvider validPaymentModeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentMode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentMode());
        self::assertEmpty($instance->payment_mode);
        $instance->setPaymentMode($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentMode()->toArray() : $instance->getPaymentMode());
        self::assertEquals($value, is_array($value) ? $instance->payment_mode->toArray() : $instance->payment_mode);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentMode());
            self::assertNotNull($instance->payment_mode);
        }
    }

    /**
     * Test invalid property "payment_mode"
     * @dataProvider invalidPaymentModeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentMode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentMode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentModeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_mode'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentModeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_mode'));
    }

    /**
     * Test property "country_of_origin_code"
     * @dataProvider validCountryOfOriginCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCountryOfOriginCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCountryOfOriginCode());
        self::assertEmpty($instance->country_of_origin_code);
        $instance->setCountryOfOriginCode($value);
        self::assertEquals($value, is_array($value) ? $instance->getCountryOfOriginCode()->toArray() : $instance->getCountryOfOriginCode());
        self::assertEquals($value, is_array($value) ? $instance->country_of_origin_code->toArray() : $instance->country_of_origin_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getCountryOfOriginCode());
            self::assertNotNull($instance->country_of_origin_code);
        }
    }

    /**
     * Test invalid property "country_of_origin_code"
     * @dataProvider invalidCountryOfOriginCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCountryOfOriginCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCountryOfOriginCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCountryOfOriginCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_country_of_origin_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCountryOfOriginCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_country_of_origin_code'));
    }

    /**
     * Test property "customs_declaration_number"
     * @dataProvider validCustomsDeclarationNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCustomsDeclarationNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCustomsDeclarationNumber());
        self::assertEmpty($instance->customs_declaration_number);
        $instance->setCustomsDeclarationNumber($value);
        self::assertEquals($value, is_array($value) ? $instance->getCustomsDeclarationNumber()->toArray() : $instance->getCustomsDeclarationNumber());
        self::assertEquals($value, is_array($value) ? $instance->customs_declaration_number->toArray() : $instance->customs_declaration_number);
        if (!empty($value)) {
            self::assertNotNull($instance->getCustomsDeclarationNumber());
            self::assertNotNull($instance->customs_declaration_number);
            self::assertLessThanOrEqual(32, is_string($instance->getCustomsDeclarationNumber()) ? mb_strlen($instance->getCustomsDeclarationNumber()) : $instance->getCustomsDeclarationNumber());
            self::assertLessThanOrEqual(32, is_string($instance->customs_declaration_number) ? mb_strlen($instance->customs_declaration_number) : $instance->customs_declaration_number);
            self::assertGreaterThanOrEqual(1, is_string($instance->getCustomsDeclarationNumber()) ? mb_strlen($instance->getCustomsDeclarationNumber()) : $instance->getCustomsDeclarationNumber());
            self::assertGreaterThanOrEqual(1, is_string($instance->customs_declaration_number) ? mb_strlen($instance->customs_declaration_number) : $instance->customs_declaration_number);
        }
    }

    /**
     * Test invalid property "customs_declaration_number"
     * @dataProvider invalidCustomsDeclarationNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCustomsDeclarationNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCustomsDeclarationNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCustomsDeclarationNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_customs_declaration_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCustomsDeclarationNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_customs_declaration_number'));
    }

    /**
     * Test property "excise"
     * @dataProvider validExciseDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExcise(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExcise());
        self::assertEmpty($instance->excise);
        $instance->setExcise($value);
        self::assertEquals($value, is_array($value) ? $instance->getExcise()->toArray() : $instance->getExcise());
        self::assertEquals($value, is_array($value) ? $instance->excise->toArray() : $instance->excise);
        if (!empty($value)) {
            self::assertNotNull($instance->getExcise());
            self::assertNotNull($instance->excise);
        }
    }

    /**
     * Test invalid property "excise"
     * @dataProvider invalidExciseDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExcise(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExcise($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExciseDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_excise'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExciseDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_excise'));
    }

    /**
     * Test property "product_code"
     * @dataProvider validProductCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testProductCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getProductCode());
        self::assertEmpty($instance->product_code);
        $instance->setProductCode($value);
        self::assertEquals($value, is_array($value) ? $instance->getProductCode()->toArray() : $instance->getProductCode());
        self::assertEquals($value, is_array($value) ? $instance->product_code->toArray() : $instance->product_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getProductCode());
            self::assertNotNull($instance->product_code);

            $value = new ProductCode($value);
            $instance->setProductCode($value);
            self::assertEquals($value, $instance->getProductCode());
            self::assertEquals($value, $instance->product_code);
        }
    }

    /**
     * Test invalid property "product_code"
     * @dataProvider invalidProductCodeDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testInvalidProductCode(mixed $value): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->getTestInstance()->productCode = $value;
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validProductCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_product_code'));
    }

    /**
     * @return array[]
     */
    public static function invalidProductCodeDataProvider(): array
    {
        return [
            [new StringObject('')],
            [true],
            [false],
            [new stdClass()],
            [Random::str(2, 96, 'GHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=`~?><:"\'')],
            [Random::str(97, 100, '0123456789ABCDEF ')],
        ];
    }

    /**
     * Test property "mark_code_info"
     * @dataProvider validMarkCodeInfoDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMarkCodeInfo(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMarkCodeInfo());
        self::assertEmpty($instance->mark_code_info);
        $instance->setMarkCodeInfo($value);
        self::assertEquals($value, is_array($value) ? $instance->getMarkCodeInfo()->toArray() : $instance->getMarkCodeInfo());
        self::assertEquals($value, is_array($value) ? $instance->mark_code_info->toArray() : $instance->mark_code_info);
        if (!empty($value)) {
            self::assertNotNull($instance->getMarkCodeInfo());
            self::assertNotNull($instance->mark_code_info);
        }
    }

    /**
     * Test invalid property "mark_code_info"
     * @dataProvider invalidMarkCodeInfoDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMarkCodeInfo(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMarkCodeInfo($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMarkCodeInfoDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_code_info'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMarkCodeInfoDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_code_info'));
    }

    /**
     * Test property "mark_mode"
     * @dataProvider validMarkModeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMarkMode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMarkMode());
        self::assertEmpty($instance->mark_mode);
        $instance->setMarkMode($value);
        self::assertEquals($value, is_array($value) ? $instance->getMarkMode()->toArray() : $instance->getMarkMode());
        self::assertEquals($value, is_array($value) ? $instance->mark_mode->toArray() : $instance->mark_mode);
        if (!empty($value)) {
            self::assertNotNull($instance->getMarkMode());
            self::assertNotNull($instance->mark_mode);
            self::assertMatchesRegularExpression("/^[0]{1}$/", $instance->getMarkMode());
            self::assertMatchesRegularExpression("/^[0]{1}$/", $instance->mark_mode);
        }
    }

    /**
     * Test invalid property "mark_mode"
     * @dataProvider invalidMarkModeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMarkMode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMarkMode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMarkModeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_mode'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMarkModeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_mode'));
    }

    /**
     * Test property "payment_subject_industry_details"
     * @dataProvider validPaymentSubjectIndustryDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentSubjectIndustryDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentSubjectIndustryDetails());
        self::assertEmpty($instance->payment_subject_industry_details);
        self::assertIsObject($instance->getPaymentSubjectIndustryDetails());
        self::assertIsObject($instance->payment_subject_industry_details);
        self::assertCount(0, $instance->getPaymentSubjectIndustryDetails());
        self::assertCount(0, $instance->payment_subject_industry_details);
        $instance->setPaymentSubjectIndustryDetails($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentSubjectIndustryDetails());
            self::assertNotNull($instance->payment_subject_industry_details);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getPaymentSubjectIndustryDetails()[$key]->toArray());
                    self::assertEquals($element, $instance->payment_subject_industry_details[$key]->toArray());
                    self::assertIsArray($instance->getPaymentSubjectIndustryDetails()[$key]->toArray());
                    self::assertIsArray($instance->payment_subject_industry_details[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getPaymentSubjectIndustryDetails()->get($key));
                    self::assertIsObject($instance->getPaymentSubjectIndustryDetails()->get($key));
                    self::assertIsObject($instance->payment_subject_industry_details->get($key));
                    self::assertIsObject($instance->getPaymentSubjectIndustryDetails());
                    self::assertIsObject($instance->payment_subject_industry_details);
                }
            }
            self::assertCount(count($value), $instance->getPaymentSubjectIndustryDetails());
            self::assertCount(count($value), $instance->payment_subject_industry_details);
        }
    }

    /**
     * Test invalid property "payment_subject_industry_details"
     * @dataProvider invalidPaymentSubjectIndustryDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentSubjectIndustryDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentSubjectIndustryDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentSubjectIndustryDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_subject_industry_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentSubjectIndustryDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_subject_industry_details'));
    }

    /**
     * Test property "supplier"
     * @dataProvider validSupplierDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSupplier(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSupplier());
        self::assertEmpty($instance->supplier);
        $instance->setSupplier($value);
        self::assertEquals($value, is_array($value) ? $instance->getSupplier()->toArray() : $instance->getSupplier());
        self::assertEquals($value, is_array($value) ? $instance->supplier->toArray() : $instance->supplier);
        if (!empty($value)) {
            self::assertNotNull($instance->getSupplier());
            self::assertNotNull($instance->supplier);
        }
    }

    /**
     * Test invalid property "supplier"
     * @dataProvider invalidSupplierDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSupplier(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSupplier($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSupplierDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_supplier'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSupplierDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_supplier'));
    }

    /**
     * Test property "agent_type"
     * @dataProvider validAgentTypeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAgentType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAgentType());
        self::assertEmpty($instance->agent_type);
        $instance->setAgentType($value);
        self::assertEquals($value, is_array($value) ? $instance->getAgentType()->toArray() : $instance->getAgentType());
        self::assertEquals($value, is_array($value) ? $instance->agent_type->toArray() : $instance->agent_type);
        if (!empty($value)) {
            self::assertNotNull($instance->getAgentType());
            self::assertNotNull($instance->agent_type);
        }
    }

    /**
     * Test invalid property "agent_type"
     * @dataProvider invalidAgentTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAgentType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAgentType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAgentTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_agent_type'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAgentTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_agent_type'));
    }

    /**
     * Test valid method "fromArray"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testFromArray(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->fromArray($value);
        $value = new ReceiptResponseItem($value);
        self::assertEquals($value->toArray(), $instance->jsonSerialize());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        $result = [];
        for ($i = 0; $i < 4; $i++) {
            $prices = $this->validAmountDataProvider();
            $array = $this->getValidDataProviderByClass($instance);
            $array[0]['amount'] = $i % 2 ? array_shift($prices[0]) : new ReceiptItemAmount(array_shift($prices[0]));
            $result[] = $array;
        }
        return $result;
    }
}
