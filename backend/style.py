STYLE = """<style>
.main-title {
    font-size: 3rem;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 1rem;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.result-container {
    background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 2rem 0;
    border-left: 5px solid #4CAF50;
}
.result-text {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1B5E20;
    margin: 0;
}
.upload-section {
    border: 2px dashed #4CAF50;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}
</style>"""

about = """
        This app uses a **MobileNetV2** neural network trained to classify waste into 6 categories:
        - 📦 Cardboard
        - 🫙 Glass  
        - 🥫 Metal
        - 📄 Paper
        - 🍶 Plastic
        - 🗑️ Trash

        The classification helps determine the correct recycling method for each item.
        """
INFO = {
    'cardboard': {
        'icon': '📦',
        'tip': 'Flatten and recycle with paper products',
        'bin': 'Paper/Cardboard recycling bin'
    },
    'glass': {
        'icon': '🫙',
        'tip': 'Clean before recycling',
        'bin': 'Glass recycling bin'
    },
    'metal': {
        'icon': '🥫',
        'tip': 'Rinse cans and containers',
        'bin': 'Metal recycling bin'
    },
    'paper': {
        'icon': '📄',
        'tip': 'Keep dry and clean',
        'bin': 'Paper recycling bin'
    },
    'plastic': {
        'icon': '🍶',
        'tip': 'Check recycling number',
        'bin': 'Plastic recycling bin'
    },
    'trash': {
        'icon': '🗑️',
        'tip': 'Not recyclable',
        'bin': 'General waste bin'
    }
}

Guidelines = """
        1. **Upload an image** of waste using the file uploader above
        2. **Click 'Classify Waste'** to analyze the image
        3. **View the result** and follow the recycling instructions
        4. **Upload another image** to classify more items

        **Tips for better results:**
        - Use clear, well-lit photos
        - Center the waste item in the image
        - Avoid cluttered backgrounds
        - Make sure the item is clearly visible
        """
