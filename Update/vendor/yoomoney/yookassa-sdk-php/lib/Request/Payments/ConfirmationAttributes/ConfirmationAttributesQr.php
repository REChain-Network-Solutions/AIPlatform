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

namespace YooKassa\Request\Payments\ConfirmationAttributes;

use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ConfirmationAttributesQr.
 *
 * Сценарий при котором пользователю необходимо просканировать QR-код.
 * От вас требуется сгенерировать QR-код, используя любой доступный инструмент, и отобразить его на странице оплаты.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код сценария подтверждения
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
 * @property string|null $return_url Адрес страницы, на которую пользователь вернется после подтверждения или отмены платежа в приложении банка.  Например, если хотите вернуть пользователя на сайт, вы можете передать URL, если в мобильное приложение — диплинк. URI должен соответствовать стандарту [RFC-3986](https://www.ietf.org/rfc/rfc3986.txt). Не более 1024 символов.  Доступно только для платежей через [СБП](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/sbp).
 * @property string|null $returnUrl Адрес страницы, на которую пользователь вернется после подтверждения или отмены платежа в приложении банка.  Например, если хотите вернуть пользователя на сайт, вы можете передать URL, если в мобильное приложение — диплинк. URI должен соответствовать стандарту [RFC-3986](https://www.ietf.org/rfc/rfc3986.txt). Не более 1024 символов.  Доступно только для платежей через [СБП](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/sbp).
 */
class ConfirmationAttributesQr extends AbstractConfirmationAttributes
{
    /**
     * Адрес страницы, на которую пользователь вернется после подтверждения или отмены платежа в приложении банка.
     *
     * Например, если хотите вернуть пользователя на сайт, вы можете передать URL, если в мобильное приложение — диплинк.
     * URI должен соответствовать стандарту [RFC-3986](https://www.ietf.org/rfc/rfc3986.txt).
     * Не более 1024 символов.
     *
     * Доступно только для платежей через [СБП](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/sbp).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 1024)]
    private ?string $_return_url = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ConfirmationType::QR);
    }

    /**
     * @return string|null Адрес страницы, на которую пользователь вернется после подтверждения или отмены платежа в приложении банка.
     */
    public function getReturnUrl(): ?string
    {
        return $this->_return_url;
    }

    /**
     * @param string|null $return_url Адрес страницы, на которую пользователь вернется после подтверждения или отмены платежа в приложении банка.
     *
     * @return self
     */
    public function setReturnUrl(?string $return_url = null): self
    {
        $this->_return_url = $this->validatePropertyValue('_return_url', $return_url);
        return $this;
    }
}
