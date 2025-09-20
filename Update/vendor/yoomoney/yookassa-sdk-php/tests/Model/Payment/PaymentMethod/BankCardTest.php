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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\BankCard;

/**
 * BankCardTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class BankCardTest extends AbstractTestCase
{
    protected BankCard $object;

    /**
     * @return BankCard
     */
    protected function getTestInstance(): BankCard
    {
        return new BankCard();
    }

    /**
     * @return void
     */
    public function testBankCardDataClassExists(): void
    {
        $this->object = $this->getMockBuilder(BankCard::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(BankCard::class));
        $this->assertInstanceOf(BankCard::class, $this->object);
    }

    /**
     * Test property "first6"
     * @dataProvider validFirst6DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFirst6(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getFirst6());
        self::assertEmpty($instance->first6);
        $instance->setFirst6($value);
        self::assertEquals($value, is_array($value) ? $instance->getFirst6()->toArray() : $instance->getFirst6());
        self::assertEquals($value, is_array($value) ? $instance->first6->toArray() : $instance->first6);
        if (!empty($value)) {
            self::assertNotNull($instance->getFirst6());
            self::assertNotNull($instance->first6);
            self::assertMatchesRegularExpression("/[0-9]{6}/", $instance->getFirst6());
            self::assertMatchesRegularExpression("/[0-9]{6}/", $instance->first6);
        }
    }

    /**
     * Test invalid property "first6"
     * @dataProvider invalidFirst6DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFirst6(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFirst6($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFirst6DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_first6'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFirst6DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_first6'));
    }

    /**
     * Test property "last4"
     * @dataProvider validLast4DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLast4(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLast4($value);
        self::assertNotNull($instance->getLast4());
        self::assertNotNull($instance->last4);
        self::assertEquals($value, is_array($value) ? $instance->getLast4()->toArray() : $instance->getLast4());
        self::assertEquals($value, is_array($value) ? $instance->last4->toArray() : $instance->last4);
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->getLast4());
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->last4);
    }

    /**
     * Test invalid property "last4"
     * @dataProvider invalidLast4DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLast4(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setLast4($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLast4DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_last4'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLast4DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_last4'));
    }

    /**
     * Test property "expiry_year"
     * @dataProvider validExpiryYearDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiryYear(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setExpiryYear($value);
        self::assertNotNull($instance->getExpiryYear());
        self::assertNotNull($instance->expiry_year);
        self::assertEquals($value, is_array($value) ? $instance->getExpiryYear()->toArray() : $instance->getExpiryYear());
        self::assertEquals($value, is_array($value) ? $instance->expiry_year->toArray() : $instance->expiry_year);
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->getExpiryYear());
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->expiry_year);
    }

    /**
     * Test invalid property "expiry_year"
     * @dataProvider invalidExpiryYearDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiryYear(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiryYear($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiryYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_year'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiryYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_year'));
    }

    /**
     * Test property "expiry_month"
     * @dataProvider validExpiryMonthDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiryMonth(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setExpiryMonth($value);
        self::assertNotNull($instance->getExpiryMonth());
        self::assertNotNull($instance->expiry_month);
        self::assertEquals($value, is_array($value) ? $instance->getExpiryMonth()->toArray() : $instance->getExpiryMonth());
        self::assertEquals($value, is_array($value) ? $instance->expiry_month->toArray() : $instance->expiry_month);
        self::assertMatchesRegularExpression("/^(0?[1-9]|1[0-2])$/", $instance->getExpiryMonth());
        self::assertMatchesRegularExpression("/^(0?[1-9]|1[0-2])$/", $instance->expiry_month);
    }

    /**
     * Test invalid property "expiry_month"
     * @dataProvider invalidExpiryMonthDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiryMonth(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiryMonth($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiryMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_month'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiryMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_month'));
    }

    /**
     * Test property "card_type"
     * @dataProvider validCardTypeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCardType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCardType($value);
        self::assertNotNull($instance->getCardType());
        self::assertNotNull($instance->card_type);
        self::assertEquals($value, is_array($value) ? $instance->getCardType()->toArray() : $instance->getCardType());
        self::assertEquals($value, is_array($value) ? $instance->card_type->toArray() : $instance->card_type);
    }

    /**
     * Test invalid property "card_type"
     * @dataProvider invalidCardTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCardType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCardType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_card_type'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_card_type'));
    }

    /**
     * Test property "card_product"
     * @dataProvider validCardProductDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCardProduct(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCardProduct());
        self::assertEmpty($instance->card_product);
        $instance->setCardProduct($value);
        self::assertEquals($value, is_array($value) ? $instance->getCardProduct()->toArray() : $instance->getCardProduct());
        self::assertEquals($value, is_array($value) ? $instance->card_product->toArray() : $instance->card_product);
        if (!empty($value)) {
            self::assertNotNull($instance->getCardProduct());
            self::assertNotNull($instance->card_product);
        }
    }

    /**
     * Test invalid property "card_product"
     * @dataProvider invalidCardProductDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCardProduct(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCardProduct($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardProductDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_card_product'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardProductDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_card_product'));
    }

    /**
     * Test property "issuer_country"
     * @dataProvider validIssuerCountryDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testIssuerCountry(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getIssuerCountry());
        self::assertEmpty($instance->issuer_country);
        $instance->setIssuerCountry($value);
        self::assertEquals($value, $instance->getIssuerCountry());
        self::assertEquals($value, $instance->issuer_country);
        if (!empty($value)) {
            self::assertNotNull($instance->getIssuerCountry());
            self::assertNotNull($instance->issuer_country);
            self::assertMatchesRegularExpression("/^[A-Z]{2}$/", $instance->getIssuerCountry());
            self::assertMatchesRegularExpression("/^[A-Z]{2}$/", $instance->issuer_country);
        }
    }

    /**
     * Test invalid property "issuer_country"
     * @dataProvider invalidIssuerCountryDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidIssuerCountry(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setIssuerCountry($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIssuerCountryDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_issuer_country'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIssuerCountryDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_issuer_country'));
    }

    /**
     * Test property "issuer_name"
     * @dataProvider validIssuerNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testIssuerName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getIssuerName());
        self::assertEmpty($instance->issuer_name);
        $instance->setIssuerName($value);
        self::assertEquals($value, is_array($value) ? $instance->getIssuerName()->toArray() : $instance->getIssuerName());
        self::assertEquals($value, is_array($value) ? $instance->issuer_name->toArray() : $instance->issuer_name);
        if (!empty($value)) {
            self::assertNotNull($instance->getIssuerName());
            self::assertNotNull($instance->issuer_name);
        }
    }

    /**
     * Test invalid property "issuer_name"
     * @dataProvider invalidIssuerNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidIssuerName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setIssuerName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIssuerNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_issuer_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIssuerNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_issuer_name'));
    }

    /**
     * Test property "source"
     * @dataProvider validSourceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSource(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSource());
        self::assertEmpty($instance->source);
        $instance->setSource($value);
        self::assertEquals($value, is_array($value) ? $instance->getSource()->toArray() : $instance->getSource());
        self::assertEquals($value, is_array($value) ? $instance->source->toArray() : $instance->source);
        if (!empty($value)) {
            self::assertNotNull($instance->getSource());
            self::assertNotNull($instance->source);
        }
    }

    /**
     * Test invalid property "source"
     * @dataProvider invalidSourceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSource(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSource($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSourceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_source'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSourceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_source'));
    }
}
