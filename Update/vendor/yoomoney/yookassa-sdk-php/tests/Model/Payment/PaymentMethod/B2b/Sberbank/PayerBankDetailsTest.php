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

namespace Tests\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\PayerBankDetails;

/**
 * B2bSberbankPayerBankDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PayerBankDetailsTest extends AbstractTestCase
{
    protected PayerBankDetails $object;

    /**
     * @return PayerBankDetails
     */
    protected function getTestInstance(): PayerBankDetails
    {
        return new PayerBankDetails();
    }

    /**
     * @return void
     */
    public function testPayerBankDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(PayerBankDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PayerBankDetails::class));
        $this->assertInstanceOf(PayerBankDetails::class, $this->object);
    }

    /**
     * Test property "full_name"
     * @dataProvider validFullNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFullName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setFullName($value);
        self::assertNotNull($instance->getFullName());
        self::assertNotNull($instance->full_name);
        self::assertEquals($value, is_array($value) ? $instance->getFullName()->toArray() : $instance->getFullName());
        self::assertEquals($value, is_array($value) ? $instance->full_name->toArray() : $instance->full_name);
        self::assertLessThanOrEqual(800, is_string($instance->getFullName()) ? mb_strlen($instance->getFullName()) : $instance->getFullName());
        self::assertLessThanOrEqual(800, is_string($instance->full_name) ? mb_strlen($instance->full_name) : $instance->full_name);
    }

    /**
     * Test invalid property "full_name"
     * @dataProvider invalidFullNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFullName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFullName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFullNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFullNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_name'));
    }

    /**
     * Test property "short_name"
     * @dataProvider validShortNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testShortName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setShortName($value);
        self::assertNotNull($instance->getShortName());
        self::assertNotNull($instance->short_name);
        self::assertEquals($value, is_array($value) ? $instance->getShortName()->toArray() : $instance->getShortName());
        self::assertEquals($value, is_array($value) ? $instance->short_name->toArray() : $instance->short_name);
        self::assertLessThanOrEqual(160, is_string($instance->getShortName()) ? mb_strlen($instance->getShortName()) : $instance->getShortName());
        self::assertLessThanOrEqual(160, is_string($instance->short_name) ? mb_strlen($instance->short_name) : $instance->short_name);
    }

    /**
     * Test invalid property "short_name"
     * @dataProvider invalidShortNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidShortName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setShortName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validShortNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_short_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidShortNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_short_name'));
    }

    /**
     * Test property "address"
     * @dataProvider validAddressDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAddress(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAddress($value);
        self::assertNotNull($instance->getAddress());
        self::assertNotNull($instance->address);
        self::assertEquals($value, is_array($value) ? $instance->getAddress()->toArray() : $instance->getAddress());
        self::assertEquals($value, is_array($value) ? $instance->address->toArray() : $instance->address);
        self::assertLessThanOrEqual(500, is_string($instance->getAddress()) ? mb_strlen($instance->getAddress()) : $instance->getAddress());
        self::assertLessThanOrEqual(500, is_string($instance->address) ? mb_strlen($instance->address) : $instance->address);
    }

    /**
     * Test invalid property "address"
     * @dataProvider invalidAddressDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAddress(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAddress($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAddressDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_address'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAddressDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_address'));
    }

    /**
     * Test property "inn"
     * @dataProvider validInnDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testInn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setInn($value);
        self::assertNotNull($instance->getInn());
        self::assertNotNull($instance->inn);
        self::assertEquals($value, is_array($value) ? $instance->getInn()->toArray() : $instance->getInn());
        self::assertEquals($value, is_array($value) ? $instance->inn->toArray() : $instance->inn);
        self::assertMatchesRegularExpression("/\\d{10}|\\d{12}/", $instance->getInn());
        self::assertMatchesRegularExpression("/\\d{10}|\\d{12}/", $instance->inn);
    }

    /**
     * Test invalid property "inn"
     * @dataProvider invalidInnDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidInn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setInn($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
     * Test property "bank_name"
     * @dataProvider validBankNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBankName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBankName($value);
        self::assertNotNull($instance->getBankName());
        self::assertNotNull($instance->bank_name);
        self::assertEquals($value, is_array($value) ? $instance->getBankName()->toArray() : $instance->getBankName());
        self::assertEquals($value, is_array($value) ? $instance->bank_name->toArray() : $instance->bank_name);
        self::assertLessThanOrEqual(350, is_string($instance->getBankName()) ? mb_strlen($instance->getBankName()) : $instance->getBankName());
        self::assertLessThanOrEqual(350, is_string($instance->bank_name) ? mb_strlen($instance->bank_name) : $instance->bank_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getBankName()) ? mb_strlen($instance->getBankName()) : $instance->getBankName());
        self::assertGreaterThanOrEqual(1, is_string($instance->bank_name) ? mb_strlen($instance->bank_name) : $instance->bank_name);
    }

    /**
     * Test invalid property "bank_name"
     * @dataProvider invalidBankNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBankName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBankName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBankNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBankNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_name'));
    }

    /**
     * Test property "bank_branch"
     * @dataProvider validBankBranchDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBankBranch(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBankBranch($value);
        self::assertNotNull($instance->getBankBranch());
        self::assertNotNull($instance->bank_branch);
        self::assertEquals($value, is_array($value) ? $instance->getBankBranch()->toArray() : $instance->getBankBranch());
        self::assertEquals($value, is_array($value) ? $instance->bank_branch->toArray() : $instance->bank_branch);
        self::assertLessThanOrEqual(140, is_string($instance->getBankBranch()) ? mb_strlen($instance->getBankBranch()) : $instance->getBankBranch());
        self::assertLessThanOrEqual(140, is_string($instance->bank_branch) ? mb_strlen($instance->bank_branch) : $instance->bank_branch);
        self::assertGreaterThanOrEqual(1, is_string($instance->getBankBranch()) ? mb_strlen($instance->getBankBranch()) : $instance->getBankBranch());
        self::assertGreaterThanOrEqual(1, is_string($instance->bank_branch) ? mb_strlen($instance->bank_branch) : $instance->bank_branch);
    }

    /**
     * Test invalid property "bank_branch"
     * @dataProvider invalidBankBranchDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBankBranch(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBankBranch($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBankBranchDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_branch'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBankBranchDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_branch'));
    }

    /**
     * Test property "bank_bik"
     * @dataProvider validBankBikDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBankBik(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBankBik($value);
        self::assertNotNull($instance->getBankBik());
        self::assertNotNull($instance->bank_bik);
        self::assertEquals($value, is_array($value) ? $instance->getBankBik()->toArray() : $instance->getBankBik());
        self::assertEquals($value, is_array($value) ? $instance->bank_bik->toArray() : $instance->bank_bik);
        self::assertMatchesRegularExpression("/\\d{9}/", $instance->getBankBik());
        self::assertMatchesRegularExpression("/\\d{9}/", $instance->bank_bik);
    }

    /**
     * Test invalid property "bank_bik"
     * @dataProvider invalidBankBikDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBankBik(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBankBik($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBankBikDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_bik'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBankBikDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_bik'));
    }

    /**
     * Test property "account"
     * @dataProvider validAccountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAccount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAccount($value);
        self::assertNotNull($instance->getAccount());
        self::assertNotNull($instance->account);
        self::assertEquals($value, is_array($value) ? $instance->getAccount()->toArray() : $instance->getAccount());
        self::assertEquals($value, is_array($value) ? $instance->account->toArray() : $instance->account);
        self::assertMatchesRegularExpression("/\\d{20}/", $instance->getAccount());
        self::assertMatchesRegularExpression("/\\d{20}/", $instance->account);
    }

    /**
     * Test invalid property "account"
     * @dataProvider invalidAccountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAccount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAccount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_account'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_account'));
    }

    /**
     * Test property "kpp"
     * @dataProvider validKppDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testKpp(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getKpp());
        self::assertEmpty($instance->kpp);
        $instance->setKpp($value);
        self::assertEquals($value, is_array($value) ? $instance->getKpp()->toArray() : $instance->getKpp());
        self::assertEquals($value, is_array($value) ? $instance->kpp->toArray() : $instance->kpp);
        if (!empty($value)) {
            self::assertNotNull($instance->getKpp());
            self::assertNotNull($instance->kpp);
            self::assertMatchesRegularExpression("/\\d{9}/", $instance->getKpp());
            self::assertMatchesRegularExpression("/\\d{9}/", $instance->kpp);
        }
    }

    /**
     * Test invalid property "kpp"
     * @dataProvider invalidKppDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidKpp(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setKpp($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validKppDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_kpp'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidKppDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_kpp'));
    }
}
