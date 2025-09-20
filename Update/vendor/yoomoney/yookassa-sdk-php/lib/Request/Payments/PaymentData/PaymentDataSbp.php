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

namespace YooKassa\Request\Payments\PaymentData;

use DateTime;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodDataSbp.
 *
 * Данные для оплаты через СБП
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $qrcId Идентификатор кассовой ссылки СБП в сервисе НСПК
 * @property string $qrc_id Идентификатор кассовой ссылки СБП в сервисе НСПК
 * @property string $expiresAt Срок действия Qr-кода — дата и время, до которых можно оплатить указанный платеж
 * @property string $expires_at Срок действия Qr-кода — дата и время, до которых можно оплатить указанный платеж
 */
class PaymentDataSbp extends AbstractPaymentData
{
    /**
     * Идентификатор кассовой ссылки СБП в сервисе НСПК. Пример: `AS1F001K6S85322B9E4P7I518N8LNEUT`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(min: 32)]
    #[Assert\Length(max: 32)]
    #[Assert\Regex("/^[A-Za-z0-9]{32}$/")]
    private ?string $_qrc_id = null;

    /**
     * Срок действия Qr-кода — дата и время, до которых можно оплатить указанный платеж. Передается для платежей по кассовой ссылке СБП.
     * Значение должно быть не менее 1 минуты и не более 20 минут от текущего времени.
     * По умолчанию срок действия Qr-кода составляет 5 минут.
     *
     * Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: `2024-10-18T10:51:18.139Z`
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::SBP);
    }

    /**
     * Возвращает идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @return string|null
     */
    public function getQrcId(): ?string
    {
        return $this->_qrc_id;
    }

    /**
     * Устанавливает идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @param string|null $qrcId Идентификатор кассовой ссылки СБП в сервисе НСПК
     *
     * @return self
     */
    public function setQrcId(?string $qrcId = null): self
    {
        $this->_qrc_id = $this->validatePropertyValue('_qrc_id', $qrcId);
        return $this;
    }

    /**
     * Возвращает срок действия Qr-кода.
     *
     * @return DateTime|null Срок действия Qr-кода
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Устанавливает срок действия Qr-кода.
     *
     * @param DateTime|string|null $expires_at Срок действия Qr-кода — дата и время, до которых можно оплатить выставленный счет.
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

}
