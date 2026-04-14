const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const sizes = [72, 96, 128, 144, 152, 192, 384, 512];
const iconsDir = path.join(__dirname, '../public/icons');

// Ensure icons directory exists
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

// Create SVG icon with shield design
const createSvg = (size) => `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f97316"/>
      <stop offset="100%" style="stop-color:#fbbf24"/>
    </linearGradient>
  </defs>
  <rect width="${size}" height="${size}" rx="${size * 0.2}" fill="url(#bg)"/>
  <g transform="translate(${size * 0.2}, ${size * 0.15}) scale(${size / 100})">
    <path
      d="M30 5 L5 15 L5 40 C5 55 15 70 30 75 C45 70 55 55 55 40 L55 15 Z"
      fill="none"
      stroke="white"
      stroke-width="4"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <path
      d="M20 35 L27 42 L42 27"
      fill="none"
      stroke="white"
      stroke-width="4"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </g>
</svg>
`;

async function generateIcons() {
  console.log('Generating PWA icons...');

  for (const size of sizes) {
    const svg = createSvg(size);
    const outputPath = path.join(iconsDir, `icon-${size}x${size}.png`);

    try {
      await sharp(Buffer.from(svg))
        .png()
        .toFile(outputPath);
      console.log(`Created: icon-${size}x${size}.png`);
    } catch (error) {
      console.error(`Error creating icon-${size}x${size}.png:`, error.message);
    }
  }

  // Also create favicon
  const faviconSvg = createSvg(32);
  try {
    await sharp(Buffer.from(faviconSvg))
      .png()
      .toFile(path.join(__dirname, '../public/favicon.png'));
    console.log('Created: favicon.png');
  } catch (error) {
    console.error('Error creating favicon:', error.message);
  }

  // Create apple-touch-icon
  const appleSvg = createSvg(180);
  try {
    await sharp(Buffer.from(appleSvg))
      .png()
      .toFile(path.join(iconsDir, 'apple-touch-icon.png'));
    console.log('Created: apple-touch-icon.png');
  } catch (error) {
    console.error('Error creating apple-touch-icon:', error.message);
  }

  console.log('Done!');
}

generateIcons();
