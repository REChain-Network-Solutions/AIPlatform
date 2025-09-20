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

namespace YooKassa\Model\Settings;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Me.
 *
 * Объект настроек магазина или шлюза (Me) — актуальная информация о настройках магазина или шлюза.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $accountId Идентификатор магазина или шлюза.
 * @property string $account_id Идентификатор магазина или шлюза.
 * @property string $status Статус магазина или шлюза.
 * @property bool $test Это тестовый магазин или шлюз.
 * @property FiscalizationData $fiscalization Настройки магазина для [отправки чеков в налоговую](https://yookassa.ru/developers/payment-acceptance/receipts/basics).
 * @property bool $fiscalizationEnabled Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков.
 * @property bool $fiscalization_enabled Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков.
 * @property string[]|array $paymentMethods Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина.
 * @property string[]|array $payment_methods Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина.
 * @property string|null $itn ИНН магазина (10 или 12 цифр). Присутствует, если вы запрашивали настройки магазина.
 * @property string[]|array $payoutMethods Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза.
 * @property string[]|array $payout_methods Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза.
 * @property string|null $name Название шлюза, которое отображается в личном кабинете ЮKassa. Присутствует, если вы запрашивали настройки шлюза.
 * @property AmountInterface|null $payoutBalance Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.
 * @property AmountInterface|null $payout_balance Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.
*/
class Me extends AbstractObject
{
    /** Подключен к ЮKassa, может проводить платежи или выплаты */
    public const STATUS_ENABLED = 'enabled';
    /** Не может проводить платежи или выплаты (еще не подключен, закрыт или временно не работает) */
    public const STATUS_DISABLED = 'disabled';

    /**
     * Идентификатор магазина или шлюза.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_account_id = null;

    /**
     * Статус магазина или шлюза. Возможные значения: `enabled` — подключен к ЮKassa, может проводить платежи или выплаты; `disabled` — не может проводить платежи или выплаты (еще не подключен, закрыт или временно не работает).
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(choices: [self::STATUS_ENABLED, self::STATUS_DISABLED])]
    #[Assert\Type('string')]
    private ?string $_status = null;

    /**
     * Это тестовый магазин или шлюз.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private ?bool $_test = null;

    /**
     * Настройки магазина для [отправки чеков в налоговую](https://yookassa.ru/developers/payment-acceptance/receipts/basics).
     *
     * @var FiscalizationData|null
     */
    #[Assert\Type(FiscalizationData::class)]
    private ?FiscalizationData $_fiscalization = null;

    /**
     * Устаревший параметр, который раньше использовался для определения настроек отправки чеков в налоговую. Сохранен для поддержки обратной совместимости, в новых версиях API может быть удален.  Используйте объект ~`fiscalization`, чтобы определить, какие у магазина настройки отправки чеков.
     *
     * @deprecated Устарел. Вместо него используйте объект `fiscalization`
     * @var bool|null
     */
    #[Assert\Type('bool')]
    private ?bool $_fiscalization_enabled = null;

    /**
     * Список [способов оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods#all), доступных магазину. Присутствует, если вы запрашивали настройки магазина.
     *
     * @var string[]|array|null
     */
    #[Assert\AllType('string')]
    #[Assert\Type('array')]
    private ?array $_payment_methods = null;

    /**
     * ИНН магазина (10 или 12 цифр). Присутствует, если вы запрашивали настройки магазина.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_itn = null;

    /**
     * Список способов получения выплат, доступных шлюзу. Возможные значения: `bank_card` — выплаты на банковские карты; `yoo_money` — выплаты на кошельки ЮMoney; `sbp` — выплаты через СБП.  Присутствует, если вы запрашивали настройки шлюза.
     *
     * @var string[]|array|null
     */
    #[Assert\AllType('string')]
    #[Assert\Type('array')]
    private ?array $_payout_methods = null;

    /**
     * Название шлюза, которое отображается в личном кабинете ЮKassa. Присутствует, если вы запрашивали настройки шлюза.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_name = null;

    /**
     * Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.
     *
     * @var AmountInterface|null
     */
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_payout_balance = null;

    /**
     * Возвращает идентификатор магазина или шлюза.
     *
     * @return string|null
     */
    public function getAccountId(): ?string
    {
        return $this->_account_id;
    }

    /**
     * Устанавливает идентификатор магазина или шлюза.
     *
     * @param string|int|null $account_id
     *
     * @return self
     */
    public function setAccountId(mixed $account_id = null): self
    {
        $this->_account_id = $this->validatePropertyValue('_account_id', (string)$account_id);
        return $this;
    }

    /**
     * Возвращает статус магазина или шлюза.
     *
     * @return string|null
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус магазина или шлюза.
     *
     * @param string|null $status Статус магазина или шлюза.
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает тестовый магазин или шлюз.
     *
     * @return bool|null
     */
    public function getTest(): ?bool
    {
        return $this->_test;
    }

    /**
     * Устанавливает тестовый магазин или шлюз.
     *
     * @param bool|null $test Это тестовый магазин или шлюз.
     *
     * @return self
     */
    public function setTest(mixed $test = null): self
    {
        $this->_test = $this->validatePropertyValue('_test', $test);
        return $this;
    }

    /**
     * Возвращает настройки магазина для отправки чеков в налоговую.
     *
     * @return FiscalizationData|null
     */
    public function getFiscalization(): ?FiscalizationData
    {
        return $this->_fiscalization;
    }

    /**
     * Устанавливает настройки магазина для отправки чеков в налоговую.
     *
     * @param FiscalizationData|array|null $fiscalization Настройки магазина для отправки чеков в налоговую.
     *
     * @return self
     */
    public function setFiscalization(mixed $fiscalization = null): self
    {
        $this->_fiscalization = $this->validatePropertyValue('_fiscalization', $fiscalization);
        return $this;
    }

    /**
     * Возвращает признак включенной фискализации.
     *
     * @deprecated Устарел.
     * @return bool|null
     */
    public function getFiscalizationEnabled(): ?bool
    {
        return $this->_fiscalization_enabled;
    }

    /**
     * Устанавливает признак включенной фискализации.
     *
     * @deprecated Устарел.
     * @param bool|null $fiscalization_enabled Признак включенной фискализации.
     *
     * @return self
     */
    public function setFiscalizationEnabled(mixed $fiscalization_enabled = null): self
    {
        $this->_fiscalization_enabled = $this->validatePropertyValue('_fiscalization_enabled', $fiscalization_enabled);
        return $this;
    }

    /**
     * Возвращает список способов оплаты, доступных магазину.
     *
     * @return string[]|array|null
     */
    public function getPaymentMethods(): ?array
    {
        return $this->_payment_methods;
    }

    /**
     * Устанавливает список способов оплаты, доступных магазину.
     *
     * @param string[]|array|null $payment_methods Список способов оплаты, доступных магазину.
     *
     * @return self
     */
    public function setPaymentMethods(mixed $payment_methods = null): self
    {
        $this->_payment_methods = $this->validatePropertyValue('_payment_methods', $payment_methods);
        return $this;
    }

    /**
     * Возвращает ИНН магазина.
     *
     * @return string|null
     */
    public function getItn(): ?string
    {
        return $this->_itn;
    }

    /**
     * Устанавливает ИНН магазина.
     *
     * @param string|null $itn ИНН магазина (10 или 12 цифр).
     *
     * @return self
     */
    public function setItn(?string $itn = null): self
    {
        $this->_itn = $this->validatePropertyValue('_itn', $itn);
        return $this;
    }

    /**
     * Возвращает список способов получения выплат.
     *
     * @return string[]|null
     */
    public function getPayoutMethods(): ?array
    {
        return $this->_payout_methods;
    }

    /**
     * Устанавливает список способов получения выплат.
     *
     * @param string[]|array|null $payout_methods Список способов получения выплат, доступных шлюзу.
     *
     * @return self
     */
    public function setPayoutMethods(mixed $payout_methods = null): self
    {
        $this->_payout_methods = $this->validatePropertyValue('_payout_methods', $payout_methods);
        return $this;
    }

    /**
     * Возвращает название шлюза.
     *
     * @return string|null
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает название шлюза.
     *
     * @param string|null $name Название шлюза, которое отображается в личном кабинете ЮKassa.
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает баланс вашего шлюза.
     *
     * @return AmountInterface|null
     */
    public function getPayoutBalance(): ?AmountInterface
    {
        return $this->_payout_balance;
    }

    /**
     * Устанавливает Баланс вашего шлюза.
     *
     * @param AmountInterface|array|null $payout_balance Баланс вашего шлюза. Присутствует, если вы запрашивали настройки шлюза.
     *
     * @return self
     */
    public function setPayoutBalance(mixed $payout_balance = null): self
    {
        $this->_payout_balance = $this->validatePropertyValue('_payout_balance', $payout_balance);
        return $this;
    }

}

