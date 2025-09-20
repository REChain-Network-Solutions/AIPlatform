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

namespace Tests\YooKassa\Model\Payment;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\AuthorizationDetails;

/**
 * AuthorizationDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class AuthorizationDetailsTest extends AbstractTestCase
{
    protected AuthorizationDetails $object;

    /**
     * @return AuthorizationDetails
     */
    protected function getTestInstance(): AuthorizationDetails
    {
        return new AuthorizationDetails();
    }

    /**
     * @return void
     */
    public function testAuthorizationDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(AuthorizationDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(AuthorizationDetails::class));
        $this->assertInstanceOf(AuthorizationDetails::class, $this->object);
    }

    /**
     * Test property "rrn"
     * @dataProvider validRrnDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRrn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRrn());
        self::assertEmpty($instance->rrn);
        $instance->setRrn($value);
        self::assertEquals($value, is_array($value) ? $instance->getRrn()->toArray() : $instance->getRrn());
        self::assertEquals($value, is_array($value) ? $instance->rrn->toArray() : $instance->rrn);
        if (!empty($value)) {
            self::assertNotNull($instance->getRrn());
            self::assertNotNull($instance->rrn);
        }
    }

    /**
     * Test invalid property "rrn"
     * @dataProvider invalidRrnDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRrn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRrn($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRrnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_rrn'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRrnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_rrn'));
    }

    /**
     * Test property "auth_code"
     * @dataProvider validAuthCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAuthCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAuthCode());
        self::assertEmpty($instance->auth_code);
        $instance->setAuthCode($value);
        self::assertEquals($value, is_array($value) ? $instance->getAuthCode()->toArray() : $instance->getAuthCode());
        self::assertEquals($value, is_array($value) ? $instance->auth_code->toArray() : $instance->auth_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getAuthCode());
            self::assertNotNull($instance->auth_code);
        }
    }

    /**
     * Test invalid property "auth_code"
     * @dataProvider invalidAuthCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAuthCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAuthCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAuthCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_auth_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAuthCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_auth_code'));
    }

    /**
     * Test property "three_d_secure"
     * @dataProvider validThreeDSecureDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testThreeDSecure(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setThreeDSecure($value);
        self::assertNotNull($instance->getThreeDSecure());
        self::assertNotNull($instance->three_d_secure);
        self::assertEquals($value, is_array($value) ? $instance->getThreeDSecure()->toArray() : $instance->getThreeDSecure());
        self::assertEquals($value, is_array($value) ? $instance->three_d_secure->toArray() : $instance->three_d_secure);
    }

    /**
     * Test invalid property "three_d_secure"
     * @dataProvider invalidThreeDSecureDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidThreeDSecure(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setThreeDSecure($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validThreeDSecureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_three_d_secure'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidThreeDSecureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_three_d_secure'));
    }
}
