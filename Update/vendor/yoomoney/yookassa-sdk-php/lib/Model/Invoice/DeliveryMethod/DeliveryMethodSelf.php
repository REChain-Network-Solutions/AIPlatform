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

namespace YooKassa\Model\Invoice\DeliveryMethod;

use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель DeliveryMethodSelf.
 *
 * Данные для самостоятельной доставки пользователю ссылки на счет.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа доставки счета пользователю.
 * @property string $url URL страницы счета, который необходимо передать пользователю для оплаты. Не более 2048 символов.
*/
class DeliveryMethodSelf extends AbstractDeliveryMethod
{
    /** @var int Максимальная длина идентификатора */
    public const MAX_LENGTH_URL = 2048;

    /**
     * URL страницы счета, который необходимо передать пользователю для оплаты. Не более 2048 символов.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_URL)]
    private ?string $_url = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(DeliveryMethodType::SELF);
    }

    /**
     * Возвращает url.
     *
     * @return string|null URL страницы счета, который необходимо передать пользователю для оплаты
     */
    public function getUrl(): ?string
    {
        return $this->_url;
    }

    /**
     * Устанавливает url.
     *
     * @param string|null $url URL страницы счета, который необходимо передать пользователю для оплаты. Не более 2048 символов.
     *
     * @return self
     */
    public function setUrl(?string $url = null): self
    {
        $this->_url = $this->validatePropertyValue('_url', $url);
        return $this;
    }

}

